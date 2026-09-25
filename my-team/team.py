import math
import random

from soccer import (
    TeamController, clamp, closest_to_ball, distance, normalise,
)

# Ball speed kept after a wall bounce. The engine does not publish this on
# `field`, but 0.75 is what it uses, and modelling the bounce measurably
# tightens the prediction.
WALL_RESTITUTION = 0.75

# How far ahead to roll the ball, in ticks. 20 ticks = 1 second. Long on
# purpose: the interception below takes the earliest point it can reach, so
# the far end only ever gets used for a ball nobody can cut off, where it is
# roughly where the ball will stop. Tuned by playing the seed set.
HORIZON = 120

# Lookahead used for positioning rather than interception: short, so the
# shape leans towards the play without over-committing.
SHAPE_LEAD = 10

# How far up the pitch of the ball the shape forms. Fifteen left a hole
# between the line and the ball that everything they won ran straight into;
# ten keeps the same shape near enough to cover it without giving up the
# ground in front.
SHAPE_DEPTH = 10.0

# Only test every Nth tick of the path for a meeting point. Two ticks is a
# tenth of a second, which is far finer than the difference matters, and it
# keeps a long horizon cheap enough to run every tick.
SCAN_STEP = 2

# How far a player stands off the centre circle during their restart. The
# player radius alone leaves one unit of margin, and a player at top speed
# coasts four after they stop pushing.
CIRCLE_STANDOFF = 2.0

# How far off the goal the covering defender sits.
COVER_DEPTH = 14.0

# Seconds of a player's own run to credit him with when working out
# how soon he could meet the ball.
MOMENTUM = 0.25

# How far off his line the keeper stands.
KEEPER_DEPTH = 2.0

# How long an untaken kickoff stays live. The engine does not publish it; the
# course guide says three seconds, and the ladder replays show the restart
# expiring sixty ticks after the goal. Past it the ball is anybody's, and
# waiting on it any longer froze whole matches at 0-0.
RESTART_SECONDS = 3.0

# How far outside the centre circle a run is kept while their kickoff is live.
# The foul is called on a player's centre crossing the circle, and a player
# swinging round it at speed drifts wide of the line he was sent along.
PATH_CLEARANCE = 1.5

# The engine caps a struck ball at this speed. Not published on `field`.
BALL_MAX_SPEED = 30.0

# Room a kick needs, in field units, beyond how far any opponent could have
# run by the time the ball passes him.
LANE_MARGIN = 0.5

# Passes worth making: shorter is not worth a touch, longer is rolling long
# enough for anybody to walk onto it.
PASS_MIN = 6.0
PASS_MAX = 35.0

# A pass is struck at distance / PASS_PACE, which gets it there in about a
# second. The rule of thumb for two seconds is thirty-three, and a pass that
# slow is a pass to them.
PASS_PACE = 20.0

# Never softer than this: below it the engine counts a touch, not a pass.
PASS_POWER_MIN = 0.5

# How far back towards our own goal a pass may go and still count as helping.
PASS_BACK = 5.0

# Directions a clearance may take, either side of straight up the pitch.
CLEAR_ANGLES = [math.radians(a) for a in range(-75, 76, 15)]

# How far down its path a clearance is checked. Past this it is loose in their
# half, which is where it was going anyway.
CLEAR_LENGTH = 25.0

# Room past this counts the same, and each unit of it is worth this much less
# than going forward: a clearance up the pitch with some room beats one across
# our own box with lots.
ROOM_CAP = 6.0
CLEAR_FORWARD = 4.0

# The sweeper stands this far goalside of their most advanced player, at
# least this far behind the ball, and never past this line.
SWEEP_GOALSIDE = 3.0
SWEEP_BEHIND = 12.0
SWEEP_LINE = -5.0

# How much nearer a new sweeper has to be before he takes the job over, so it
# does not change hands every tick.
SWEEP_STICKY = 5.0


class BallPath:
    """The ball's predicted track, rolled forward only as far as it is read.

    The same maths as before, kept behind a method instead of returned whole.
    Most ticks somebody meets the ball inside the first half-second, and the
    remaining five and a half were being built and thrown away every tick.
    """

    __slots__ = ("xs", "ys", "vx", "vy", "dt", "fr", "hx", "hy")

    def __init__(self, obs):
        f = obs.field
        self.dt = 1.0 / f.simulation_hz
        self.fr = f.ball_friction
        # The ball centre stops one radius short of the boundary.
        self.hx = f.width / 2 - f.ball_radius
        self.hy = f.height / 2 - f.ball_radius
        x, y = obs.ball.position
        self.vx, self.vy = obs.ball.velocity
        self.xs = [x]
        self.ys = [y]

    def grow(self, sample):
        """Roll forward until `sample` exists.

        Still a tick at a time, because a wall bounce has to be caught on the
        tick it happens; only every SCAN_STEP-th position is kept, because
        that is all anything reads.
        """
        xs, ys = self.xs, self.ys
        x, y, vx, vy = xs[-1], ys[-1], self.vx, self.vy
        dt, fr, hx, hy = self.dt, self.fr, self.hx, self.hy
        for _ in range(sample + 1 - len(xs)):
            for _ in range(SCAN_STEP):
                x, y = x + vx * dt, y + vy * dt
                vx, vy = vx * fr, vy * fr
                # Reflect back across whichever wall it went through.
                if y > hy:
                    y, vy = 2 * hy - y, -vy * WALL_RESTITUTION
                elif y < -hy:
                    y, vy = -2 * hy - y, -vy * WALL_RESTITUTION
                if x > hx:
                    x, vx = 2 * hx - x, -vx * WALL_RESTITUTION
                elif x < -hx:
                    x, vx = -2 * hx - x, -vx * WALL_RESTITUTION
            xs.append(x)
            ys.append(y)
        self.vx, self.vy = vx, vy

    def at(self, sample):
        self.grow(sample)
        return self.xs[sample], self.ys[sample]


def reach_table(obs, ticks=HORIZON):
    """How far a player can run by each point on the path.

    One entry per stored point, so it lines up with `BallPath` and is indexed
    by the same number rather than by a tick.

    Players accelerate from a standstill, so distance / max_speed alone is
    optimistic over short runs; this adds the ramp up to top speed. It depends
    only on the physics, so it is built once a match and shared by everyone.
    """
    f = obs.field
    ramp = f.max_speed / f.acceleration            # seconds spent accelerating
    ramp_gap = 0.5 * f.acceleration * ramp * ramp  # ground covered doing it
    # Being within kick range counts as having arrived.
    slack = f.kick_range * 0.5

    table = []
    for i in range(ticks // SCAN_STEP + 1):
        seconds = i * SCAN_STEP / f.simulation_hz
        if seconds <= ramp:
            gap = 0.5 * f.acceleration * seconds * seconds
        else:
            gap = ramp_gap + (seconds - ramp) * f.max_speed
        table.append(gap + slack)
    return table


def earliest(path, px, py, reach, limit):
    """Earliest sample this player reaches, or -1 if none before `limit`.

    `limit` is the best time a team-mate has already managed. A player who
    cannot beat it will not be the chaser, so the scan stops there rather than
    running the whole horizon out for somebody losing the race. Saying -1 for
    a miss rather than `limit` matters: running out of scan is not a tie.

    Compares squared distances against `reach` rather than working out an
    arrival time per point: same answer, no square roots.
    """
    xs, ys = path.xs, path.ys
    for i in range(limit):
        if i >= len(xs):
            path.grow(min(i + 8, limit - 1))
        dx = px - xs[i]
        dy = py - ys[i]
        r = reach[i]
        if dx * dx + dy * dy <= r * r:
            return i
    return -1


def roll_table(obs, ticks=HORIZON):
    """How far a struck ball has rolled by each stored point, per unit of the
    speed it left the boot at.

    Friction takes the same share of the speed every tick, so the distance is
    a geometric series and the launch speed only scales it. Like the reach
    table it depends only on the physics, so it is built once a match.
    """
    f = obs.field
    dt = 1.0 / f.simulation_hz
    fr = f.ball_friction
    return [dt * (1.0 - fr ** (i * SCAN_STEP)) / (1.0 - fr)
            for i in range(ticks // SCAN_STEP + 1)]


def their_restart(obs, started):
    """Whether their kickoff is live, with the ball still on the centre spot.

    While it is, the centre circle belongs to them: standing in it is the only
    foul in the game, and it costs that player about four seconds walking back
    from a touchline.

    `started` is the tick the restart began on. A kickoff nobody takes expires,
    and possession stays with them while the ball sits on the spot, so asking
    only about the ball said "their restart" for the rest of the match.
    """
    bx, by = obs.ball.position
    live = RESTART_SECONDS * obs.field.simulation_hz
    return (obs.ball.controlling_team == 1 and bx * bx + by * by < 1.0
            and obs.tick - started <= live)


def around_circle(obs, position, spot):
    """Where to head for so the run to `spot` never cuts through the circle.

    Moving the target out of the circle is not enough: a player on the far
    side of it runs straight across the middle to get there. If the straight
    line comes too close, head for the point where it would just touch the
    circle instead, on the side the target is; re-asked every tick, that walks
    him round the edge.
    """
    r = obs.field.centre_circle_radius + PATH_CLEARANCE
    px, py = position
    d = math.hypot(px, py)
    sx, sy = spot
    if d <= r:
        # Already too close: out and round at once, towards the target's side.
        if d < 1e-6:
            return (-2.0 * r, 0.0)
        ox, oy = px / d, py / d
        side = 1.0 if ox * sy - oy * sx >= 0.0 else -1.0
        return (px + (ox - side * oy) * r, py + (oy + side * ox) * r)

    dx, dy = sx - px, sy - py
    length2 = dx * dx + dy * dy
    t = 0.0
    if length2 > 1e-9:
        t = clamp(-(px * dx + py * dy) / length2, 0.0, 1.0)
    cx, cy = px + t * dx, py + t * dy
    if cx * cx + cy * cy >= r * r:
        return spot

    base = math.atan2(py, px)
    half = math.acos(r / d)
    goal = math.atan2(sy, sx)
    # Whichever tangent point turns him the shorter way round to the target.
    a = min((base + half, base - half),
            key=lambda a: abs(math.remainder(goal - a, math.tau)))
    return (r * math.cos(a), r * math.sin(a))


def launch(obs, direction, power):
    """The ball's velocity once a kick of this power lands.

    The impulse is added to whatever the ball was already doing, and the
    result capped, exactly as the engine does it.
    """
    vx, vy = obs.ball.velocity
    p = power * obs.field.kick_impulse
    vx += direction[0] * p
    vy += direction[1] * p
    speed = math.hypot(vx, vy)
    if speed > BALL_MAX_SPEED:
        vx, vy = vx * BALL_MAX_SPEED / speed, vy * BALL_MAX_SPEED / speed
    return vx, vy


def pass_power(gap):
    """How hard to strike a pass that has `gap` to travel."""
    return clamp(gap / PASS_PACE, PASS_POWER_MIN, 1.0)


def by_distance(point):
    """Sort key putting players nearest `point` first, without square roots."""
    x, y = point
    return lambda p: (p.position[0] - x) ** 2 + (p.position[1] - y) ** 2


def step_towards(start, end, length):
    """`start` moved `length` towards `end`; straight up the pitch if the two
    coincide and there is no direction to take."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    gap = math.hypot(dx, dy)
    if gap > 1e-6:
        return (start[0] + dx / gap * length, start[1] + dy / gap * length)
    return (start[0] + length, start[1])


def move(player, spot):
    """Run at `spot` flat out.

    A plain dict, not a PlayerAction: the engine reads either, and the typed
    one is a frozen dataclass built five times a tick.
    """
    px, py = player.position
    dx, dy = spot[0] - px, spot[1] - py
    gap = math.hypot(dx, dy)
    return {"movement": (dx / gap, dy / gap) if gap > 1e-9 else (0.0, 0.0)}


def kick(direction, power):
    """Strike the ball from a standstill."""
    return {"movement": (0.0, 0.0), "kick_direction": direction,
            "kick_power": power}


def outside_circle(obs, spot):
    """The same target, pushed clear of the centre circle."""
    edge = (obs.field.centre_circle_radius + obs.field.player_radius
            + CIRCLE_STANDOFF)
    gap = distance(spot, (0.0, 0.0))
    if gap >= edge:
        return spot
    if gap < 1e-6:
        return (-edge, 0.0)      # dead centre: back off into our own half
    return (spot[0] * edge / gap, spot[1] * edge / gap)


def kick_aim(obs, target, power):
    """The direction to kick so the ball ends up travelling at `target`.

    A kick adds to the ball's velocity rather than replacing it, so a ball
    that is already rolling leaves at the sum of the two and not along the
    line it was struck. Aimed at the middle of a goal there is enough of the
    mouth either side to absorb that; aimed at a corner there is none, and the
    same shot that used to be saved now misses entirely.

    So solve for it instead. We want ``v + p*d`` to point at the target for
    some unit ``d``, i.e. ``p*d = k*u - v`` with ``u`` the unit vector to the
    target and ``k`` the speed the ball leaves at. Requiring ``d`` to be a unit
    vector makes that a quadratic in ``k``, and the larger root is the one that
    sends the ball forwards.

    Falls back to the straight line when the ball is crossing too fast for a
    kick of this power to redirect it, which is then the best there is.
    """
    bx, by = obs.ball.position
    ux, uy = normalise((target[0] - bx, target[1] - by))
    vx, vy = obs.ball.velocity
    p = power * obs.field.kick_impulse

    along = ux * vx + uy * vy
    disc = along * along - (vx * vx + vy * vy) + p * p
    if disc <= 0.0:
        return (ux, uy)
    k = along + math.sqrt(disc)
    return normalise((k * ux - vx, k * uy - vy))


class MyTeam(TeamController):
    name = "Connor Pace"
    version = "13"

    # How far towards a post a shot is aimed, as a fraction of the goal mouth.
    # 1.0 is the inside of the post itself, which is missed about as often as
    # it is hit; this keeps the ball inside the frame while still asking the
    # keeper to cover the full width of the goal.
    SHOT_INSET = 0.85

    # Spread applied inside the half of the goal the shot has picked, so two
    # shots from the same place do not go to the same spot.
    SHOT_JITTER = 0.22

    # All rebuilt by `reset`; the defaults only keep a controller that never
    # had one called from raising on its first tick.
    _rng = random.Random(0)
    _reach = None
    _roll = None
    _restart = 0
    _score = (0, 0)
    _sweeper = None

    def reset(self, seed):
        """Per-match state.

        Seeding from the match seed rather than the clock is what keeps a
        varied shot from costing reproducibility: the same seed still replays
        into the same match.
        """
        self._rng = random.Random(seed)
        self._reach = None
        self._roll = None
        # The tick the current restart began on: the opening kickoff, then
        # every goal.
        self._restart = 0
        self._score = (0, 0)
        self._sweeper = None

    def their_keeper(self, obs):
        """Whoever is guarding their goal, whatever slot they keep them in.

        Asked as "nearest opponent to the goal they defend" rather than read
        off a fixed id, so a side that does not use slot 0 as a keeper — or
        that has lost that player to a foul — still gets read correctly.
        """
        return obs.closest_opponent_to(obs.opponent_goal)

    def aim(self, obs, shooter):
        """The half of their goal their keeper is not standing in.

        Spreading a shot across the middle of the mouth still aims it at the
        one player paid to stand there. The keeper can only be on one side of
        the goal at a time, so the shot goes to the other one: the corner
        furthest from where they will be when the ball arrives, pulled in off
        the post so it stays inside the frame.
        """
        f = obs.field
        mouth = f.goal_width / 2 - f.ball_radius
        edge = mouth * self.SHOT_INSET
        goal_x = obs.opponent_goal[0]

        keeper = self.their_keeper(obs)
        if keeper is None:
            return (goal_x, self._rng.uniform(-edge, edge))

        # Where they will be when it gets there, not where they are now. A
        # keeper already moving carries on doing so while the ball is in
        # flight; ignoring that aims at the space they are in the act of
        # leaving.
        gap = distance(shooter.position, (goal_x, keeper.position[1]))
        flight = obs.ticks_to_cover(gap, 1.0) / f.simulation_hz
        keeper_y = keeper.position[1] + keeper.velocity[1] * flight

        # Whichever post they are further from. Ties, and a keeper standing
        # dead centre, go to a random side rather than always the same one.
        if keeper_y > 0.0:
            target = -edge
        elif keeper_y < 0.0:
            target = edge
        else:
            target = self._rng.choice((-edge, edge))

        # Jitter inwards only, so varying the shot never walks it past a post.
        target -= target * self._rng.random() * self.SHOT_JITTER
        return (goal_x, target)

    def marking(self, obs, defenders):
        """One opponent each, the most dangerous taken first.

        Asking each defender for the opponent nearest to *them* leaves two
        standing on the same man while the dangerous one runs free. Walking
        the threats instead - nearest the goal we defend first - and giving
        each of them the closest defender still unassigned fixes both, and
        leaves their keeper unmarked, which is where he is least trouble.
        """
        free = list(defenders)
        pairs = {}
        for them in sorted(obs.opponents, key=lambda o: o.position[0]):
            if not free:
                break
            mine = min(free, key=lambda p: distance(p.position, them.position))
            free.remove(mine)
            pairs[mine.id] = them
        return pairs

    def lane_room(self, obs, velocity, length, ignore=None):
        """The least room any opponent leaves a ball struck at `velocity`.

        Room is how much further an opponent is from each point of the run
        than he could have run by the time the ball gets there. Below zero,
        somebody gets a foot to it first. This is what striking everything at
        goal from anywhere kept losing to: more than half of those kicks were
        next touched by them inside a second, and a forward standing a few
        yards in front of the kick sent it straight back past our keeper.

        Checked as far as `length`, or a wall, whichever the ball gets to
        first. `ignore` is an opponent id to leave out.
        """
        speed = math.hypot(velocity[0], velocity[1])
        if speed < 1e-6:
            return -math.inf
        ux, uy = velocity[0] / speed, velocity[1] / speed
        bx, by = obs.ball.position
        hx = obs.field.width / 2
        hy = obs.field.height / 2
        roll, reach = self._roll, self._reach

        points = []
        for i in range(1, len(roll)):
            s = speed * roll[i]
            px, py = bx + ux * s, by + uy * s
            points.append((px, py, reach[i]))
            if s >= length or abs(px) > hx or abs(py) > hy:
                break

        room = math.inf
        for them in obs.opponents:
            if them.id == ignore:
                continue
            # Where their own run carries them, as for our players.
            ox = them.position[0] + them.velocity[0] * MOMENTUM
            oy = them.position[1] + them.velocity[1] * MOMENTUM
            for px, py, r in points:
                dx, dy = ox - px, oy - py
                gap = math.sqrt(dx * dx + dy * dy) - r
                if gap < room:
                    room = gap
        return room - LANE_MARGIN

    def shot(self, obs, shooter):
        """A shot, if nobody but their keeper can get in front of it.

        The keeper is left out: the aim already goes to the side he is not
        on, and he is the one opponent a shot is meant to beat.
        """
        target = self.aim(obs, shooter)
        direction = kick_aim(obs, target, 1.0)
        velocity = launch(obs, direction, 1.0)
        if velocity[0] <= 0.0:
            return None
        keeper = self.their_keeper(obs)
        length = distance(obs.ball.position, target)
        if self.lane_room(obs, velocity, length,
                          keeper.id if keeper else None) > 0.0:
            return direction
        return None

    def pass_to(self, obs, passer):
        """A pass to the nearest team-mate it would help, if there is one.

        Helps means the ball gets to him before any of them, it goes
        somewhere rather than back towards our own goal, and it is neither a
        tap to a man alongside nor long enough for anybody to run onto.
        Nearest first, and the first that passes is the one taken.
        """
        bx, by = obs.ball.position
        hx = obs.field.width / 2 - 1.0
        hy = obs.field.height / 2 - 1.0
        mates = [p for p in obs.my_players if p.id != 0 and p.id != passer.id]
        mates.sort(key=by_distance(obs.ball.position))

        for mate in mates:
            # Lead him: where he will be when the ball arrives, settled over
            # two rounds as the course guide describes.
            mx, my = mate.position
            tx, ty = mx, my
            for _ in range(2):
                gap = math.hypot(tx - bx, ty - by)
                flight = gap / (pass_power(gap) * obs.field.kick_impulse)
                tx = clamp(mx + mate.velocity[0] * flight, -hx, hx)
                ty = clamp(my + mate.velocity[1] * flight, -hy, hy)

            gap = math.hypot(tx - bx, ty - by)
            if not PASS_MIN <= gap <= PASS_MAX or tx < bx - PASS_BACK:
                continue
            power = pass_power(gap)
            direction = kick_aim(obs, (tx, ty), power)
            if self.lane_room(obs, launch(obs, direction, power), gap) > 0.0:
                return direction, power
        return None

    def clearance(self, obs):
        """Full power into whichever direction has the most room, leaning
        up the pitch, and never straight into the nearest wall."""
        bx, by = obs.ball.position
        hy = obs.field.height / 2 - 1.0
        best, best_score = (1.0, 0.0), -math.inf
        for angle in CLEAR_ANGLES:
            ux, uy = math.cos(angle), math.sin(angle)
            if abs(by + uy * 10.0) > hy:
                continue
            direction = kick_aim(obs, (bx + ux * 30.0, by + uy * 30.0), 1.0)
            room = self.lane_room(obs, launch(obs, direction, 1.0),
                                  CLEAR_LENGTH)
            score = min(room, ROOM_CAP) + CLEAR_FORWARD * ux
            if score > best_score:
                best, best_score = direction, score
        return best

    def strike(self, obs, player):
        """What to do with the ball once it is at his feet.

        Shoot if the lane is open, and pass to the nearest team-mate if that
        helps. Failing both, shoot anyway in their half, where a blocked shot
        still leaves the scramble in front of their goal; in ours, clear it
        into space rather than into a body. Clearing in their half as well
        cost goals against every side that sits deep.
        """
        direction = self.shot(obs, player)
        if direction is not None:
            return kick(direction, 1.0)
        found = self.pass_to(obs, player)
        if found is not None:
            return kick(*found)
        if obs.ball.position[0] > 0.0:
            return kick(kick_aim(obs, self.aim(obs, player), 1.0), 1.0)
        return kick(self.clearance(obs), 1.0)

    def act(self, obs):
        # The one method the engine calls.
        self.note_restart(obs)

        # Both tables depend only on the physics, and those do not change
        # while a match is running, so they are built on the first tick and
        # reused for the rest. Rebuilding the reach table every tick was about
        # a fifth of the whole decision, and the decision time is what
        # separates teams on equal points.
        if self._reach is None:
            self._reach = reach_table(obs)
            self._roll = roll_table(obs)

        if closest_to_ball(obs):
            return self.on_the_ball(obs)
        return self.off_the_ball(obs)

    def note_restart(self, obs):
        """Keep track of the tick the current restart began on.

        The goal event carries the tick it went in on, which is when the
        restart clock starts; a score that moved without one still counts, a
        tick late.
        """
        restarted = False
        for event in obs.events:
            if event["kind"] == "goal":
                self._restart = event["tick"]
                restarted = True
        score = tuple(obs.score)
        if score != self._score:
            if not restarted:
                self._restart = obs.tick - 1
            self._score = score

    def run_to(self, actions, obs, player, spot, blocked):
        """Send a player at a target, keeping out of their kickoff circle."""
        if blocked:
            spot = around_circle(obs, player.position,
                                 outside_circle(obs, spot))
        actions[player.id] = move(player, spot)

    def chase(self, obs, path):
        """Who meets the ball soonest, and where they meet it.

        Only the winner's meeting point is ever used, so once somebody has a
        time nobody else is scanned past it. Nearest the ball first, because
        that player usually wins and makes the cap small straight away. The
        answer is the one a full scan per player gives, ties included.
        """
        reach = self._reach
        last = len(reach) - 1

        runners = [p for p in obs.my_players if p.id != 0] or list(obs.my_players)
        runners.sort(key=by_distance(obs.ball.position))

        best_id, best_t = None, last + 1
        for player in runners:
            # Where his own momentum carries him, not where he is standing:
            # the reach table starts everyone from rest, which undersells a
            # player already running at the ball and oversells one running
            # away from it.
            px, py = player.position
            vx, vy = player.velocity
            px += vx * MOMENTUM
            py += vy * MOMENTUM
            # best_t + 1, not best_t: a player who ties the leader has to be
            # seen to tie, or the tie-break below never runs and the lower id
            # loses a race the old full scan gave it.
            found = earliest(path, px, py, reach, min(best_t + 1, last + 1))
            if found < 0:
                continue                        # cannot beat the leader
            if found < best_t or (found == best_t and player.id < best_id):
                best_id, best_t = player.id, found
        if best_id is None:
            # Nobody gets there inside the horizon. The lowest id heads for the
            # end of the path, which is roughly where the ball will stop.
            best_id = min(p.id for p in runners)
        return best_id, path.at(min(best_t, last))

    def keep_goal(self, actions, obs, keeper):
        """Hold the line where a shot would arrive, and clear anything close.

        Standing in the way is not enough: the ball keeps most of its speed
        off a body, so a shot the keeper merely blocks carries on into the
        net. Kicking it is what turns a block into a save.
        """
        if obs.can_kick(keeper.id):
            # To the nearest team-mate if the pass is on, and otherwise
            # upfield into space. Hard and straight up the middle was how a
            # forward standing in front of him scored off his clearances.
            found = self.pass_to(obs, keeper)
            if found is not None:
                actions[keeper.id] = kick(*found)
            else:
                actions[keeper.id] = kick(self.clearance(obs), 1.0)
            return
        # Stay deep either way. Coming out to narrow the angle only looks
        # right: a block is not a save here, because the ball keeps most of
        # its speed off a body. The clearance above is the save, and you have
        # to be on the ball to make it.
        f = obs.field
        mouth = f.goal_width / 2
        line_x = obs.my_goal[0] + KEEPER_DEPTH
        bx, by = obs.ball.position
        vx, vy = obs.ball.velocity

        # Where the ball crosses his line, if it is coming at all. Friction
        # scales both axes by the same factor, so the whole roll is one
        # parameter: the ball is at (bx + vx * s, by + vy * s) for an s that
        # runs from zero to dt / (1 - friction) and no further. Solving the x
        # of that for the keeper's line gives the s it arrives at, and the y
        # falls straight out of it. No walking the path, and nothing to tune.
        spot_y = None
        if vx < -1e-6:
            roll = (1.0 / f.simulation_hz) / (1.0 - f.ball_friction)
            s = (line_x - bx) / vx
            if 0.0 < s <= roll:
                spot_y = by + vy * s

        if spot_y is None:
            # Nothing coming at him. Sit on the line from the ball to the
            # middle of the goal, which is the middle of every shot it could
            # take from there. Two units off his line that is a long way short
            # of the ball's own y, and copying the ball's y instead left the
            # whole far half of the goal open to a shot from the wing.
            span = bx - obs.my_goal[0]
            spot_y = by * KEEPER_DEPTH / span if span > 1e-6 else by

        actions[keeper.id] = move(keeper, (line_x, clamp(spot_y, -mouth, mouth)))

    def on_the_ball(self, obs):
        actions = {}
        path = BallPath(obs)
        # Whoever gets there soonest, which is not always whoever is nearest.
        chaser_id, meet = self.chase(obs, path)
        # Where the ball will be while the shape is being taken up.
        lead = path.at(SHAPE_LEAD // SCAN_STEP)
        blocked = their_restart(obs, self._restart)
        sweeper_id, sweep, lanes = self.attack_shape(obs, chaser_id, lead[0])

        # Every one of your players goes through this loop exactly once and
        # leaves it with exactly one action.
        for player in obs.my_players:
            if player.id == 0:
                # Keeper: hold the goal line, slide across with the ball.
                self.keep_goal(actions, obs, player)

            elif player.id == chaser_id:
                # The one player meeting the ball; nobody else follows it.
                if obs.can_kick(player.id):
                    actions[player.id] = self.strike(obs, player)
                else:
                    # Run to where the ball is going, not where it is.
                    self.run_to(actions, obs, player, meet, blocked)

            elif player.id == sweeper_id:
                self.run_to(actions, obs, player, sweep, blocked)

            else:
                spot = (lead[0] + SHAPE_DEPTH, lanes[player.id])
                self.run_to(actions, obs, player, spot, blocked)

        return actions

    def attack_shape(self, obs, chaser_id, ball_x):
        """Who sweeps, where, and the lane each of the others takes.

        One man stays back. With all four ahead of the ball, the sides that
        beat us left two forwards on our box and scored off whatever came out
        of it, with one or two of ours goalside at best.

        The rest spread out ahead of the ball, one lane each, handed out in
        the order they already stand across the pitch so nobody has to cross
        anybody to get to his. Centred on the pitch, so the shape does not sit
        in one channel with its widest man on a touchline.
        """
        spare = [p for p in obs.my_players if p.id != 0 and p.id != chaser_id]
        sweeper_id, sweep = None, None
        if len(spare) > 1:
            sweep = self.sweeper_spot(obs, ball_x)
            sweeper_id = min(spare, key=lambda p: distance(p.position, sweep)
                             - (SWEEP_STICKY if p.id == self._sweeper else 0.0)).id
            self._sweeper = sweeper_id

        runners = sorted((p for p in spare if p.id != sweeper_id),
                         key=lambda p: p.position[1])
        width = obs.field.height * (0.25 if len(runners) < 4 else 0.2)
        lanes = {p.id: (i - (len(runners) - 1) * 0.5) * width
                 for i, p in enumerate(runners)}
        return sweeper_id, sweep, lanes

    def sweeper_spot(self, obs, ball_x):
        """Goalside of their most advanced player, and well behind the ball.

        Their keeper is left out of "most advanced", or a side with nobody
        forward would have the sweeper marking the far goal.
        """
        keeper = self.their_keeper(obs)
        outfield = [o for o in obs.opponents
                    if keeper is None or o.id != keeper.id] or obs.opponents
        them = min(outfield, key=lambda o: o.position[0])
        x, y = step_towards(them.position, obs.my_goal, SWEEP_GOALSIDE)
        x = min(x, ball_x - SWEEP_BEHIND, SWEEP_LINE)
        return (max(x, obs.my_goal[0] + COVER_DEPTH), y)

    def off_the_ball(self, obs):
        actions = {}
        path = BallPath(obs)
        chaser_id, meet = self.chase(obs, path)
        blocked = their_restart(obs, self._restart)

        # Whoever is next-nearest the ball supports the chaser instead of
        # picking up a man. One player at the ball wins the race and then
        # loses every loose ball that comes off it, and a loose ball in their
        # half is the cheapest shot on offer.
        bx, by = obs.ball.position
        rest = [p for p in obs.my_players if p.id != 0 and p.id != chaser_id]
        second_id = None
        if len(rest) > 1:
            second_id = min(rest, key=by_distance(obs.ball.position)).id
            rest = [p for p in rest if p.id != second_id]
        # One of the rest drops in front of the keeper instead of taking a
        # man. Marking stands a defender beside an opponent, which is no use
        # against a shot: it leaves the line to our goal open and the keeper
        # alone on it. Whoever is deepest already takes the job.
        cover_id = None
        if len(rest) > 1:
            cover_id = min(rest, key=lambda p: p.position[0]).id
            rest = [p for p in rest if p.id != cover_id]
        pairs = self.marking(obs, rest)

        for player in obs.my_players:
            if player.id == 0:
                self.keep_goal(actions, obs, player)

            elif player.id == chaser_id:
                # Cut the ball off rather than following it around.
                self.run_to(actions, obs, player, meet, blocked)

            elif player.id == second_id:
                # Second man. He stands off the ball rather than on top of the
                # chaser, and goalside of it, so a ball that squirts loose runs
                # to him and not to them.
                self.run_to(actions, obs, player, (bx - 6.0, by), blocked)

            elif player.id == cover_id:
                # On the line from the ball to the goal we defend, so a shot
                # has to beat a body before it reaches the keeper.
                spot = step_towards(obs.my_goal, obs.ball.position, COVER_DEPTH)
                self.run_to(actions, obs, player, spot, blocked)

            else:
                # Mark goalside: stand between your man and the goal you
                # defend, which is always at -x. One man each, picked above.
                them = pairs.get(player.id)
                if them is None:
                    spot = obs.my_goal
                else:
                    spot = (them.position[0] - 3.0, them.position[1])
                self.run_to(actions, obs, player, spot, blocked)

        return actions
