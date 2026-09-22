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
# coasts four after they stop pushing, which is where the fouls came from.
CIRCLE_STANDOFF = 2.0

# How far off the goal the covering defender sits.
COVER_DEPTH = 14.0

# Seconds of a player's own run to credit him with when working out
# how soon he could meet the ball.
MOMENTUM = 0.25

# How far off his line the keeper stands.
KEEPER_DEPTH = 2.0


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

    One entry per stored point, so it lines up with `ball_path` and is indexed
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


def their_restart(obs):
    """Whether their kickoff is live, with the ball still on the centre spot.

    While it is, the centre circle belongs to them: standing in it is the only
    foul in the game, and it costs that player about four seconds walking back
    from a touchline.
    """
    bx, by = obs.ball.position
    return obs.ball.controlling_team == 1 and bx * bx + by * by < 1.0


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
    version = "12"

    # How far towards a post a shot is aimed, as a fraction of the goal mouth.
    # 1.0 is the inside of the post itself, which is missed about as often as
    # it is hit; this keeps the ball inside the frame while still asking the
    # keeper to cover the full width of the goal.
    SHOT_INSET = 0.85

    # Spread applied inside the half of the goal the shot has picked, so two
    # shots from the same place do not go to the same spot.
    SHOT_JITTER = 0.22

    # Both are rebuilt by `reset`; the defaults only keep a controller that
    # never had one called from raising on its first tick.
    _rng = random.Random(0)
    _reach = None

    def reset(self, seed):
        """Per-match state.

        Seeding from the match seed rather than the clock is what keeps a
        varied shot from costing reproducibility: the same seed still replays
        into the same match.
        """
        self._rng = random.Random(seed)
        self._reach = None

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

    def act(self, obs):
        # The one method the engine calls. Everything starts with deciding
        # what kind of moment this is; this team asks the simplest question
        # there is, and you should expect to outgrow it.
        if closest_to_ball(obs):
            return self.on_the_ball(obs)
        return self.off_the_ball(obs)

    def run_to(self, actions, obs, player, spot, blocked):
        """Send a player at a target, keeping out of their kickoff circle."""
        if blocked:
            spot = outside_circle(obs, spot)
        # A plain dict, not a PlayerAction: the engine reads either, and the
        # typed one is a frozen dataclass built five times a tick.
        px, py = player.position
        dx, dy = spot[0] - px, spot[1] - py
        gap = math.hypot(dx, dy)
        actions[player.id] = {
            "movement": (dx / gap, dy / gap) if gap > 1e-9 else (0.0, 0.0)}

    def chase(self, obs, path):
        """Who meets the ball soonest, and where they meet it.

        Only the winner's meeting point is ever used, so once somebody has a
        time nobody else is scanned past it. Nearest the ball first, because
        that player usually wins and makes the cap small straight away. The
        answer is the one a full scan per player gives, ties included.

        The reach table depends only on the physics, and those do not change
        while a match is running, so it is built on the first tick and reused
        for the rest. Rebuilding it every tick was about a fifth of the whole
        decision, and the decision time is what separates teams on equal
        points.
        """
        reach = self._reach
        if reach is None:
            reach = self._reach = reach_table(obs)
        last = len(reach) - 1

        bx, by = obs.ball.position
        runners = [p for p in obs.my_players if p.id != 0] or list(obs.my_players)
        runners.sort(key=lambda p: (p.position[0] - bx) ** 2
                     + (p.position[1] - by) ** 2)

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
            # Clear it upfield, hard, and away from the middle.
            aim = (obs.opponent_goal[0], obs.ball.position[1] * 3.0)
            actions[keeper.id] = {"movement": (0.0, 0.0),
                                  "kick_direction": kick_aim(obs, aim, 1.0),
                                  "kick_power": 1.0}
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

        spot = (line_x, clamp(spot_y, -mouth, mouth))
        kx, ky = keeper.position
        dx, dy = spot[0] - kx, spot[1] - ky
        gap = math.hypot(dx, dy)
        actions[keeper.id] = {
            "movement": (dx / gap, dy / gap) if gap > 1e-9 else (0.0, 0.0)}

    def on_the_ball(self, obs):
        actions = {}
        path = BallPath(obs)
        # Whoever gets there soonest, which is not always whoever is nearest.
        chaser_id, meet = self.chase(obs, path)
        # Where the ball will be while the shape is being taken up.
        lead = path.at(SHAPE_LEAD // SCAN_STEP)
        blocked = their_restart(obs)

        # Every one of your players goes through this loop exactly once and
        # leaves it with exactly one action.
        for player in obs.my_players:
            if player.id == 0:
                # Keeper: hold the goal line, slide across with the ball.
                self.keep_goal(actions, obs, player)

            elif player.id == chaser_id:
                # The one player meeting the ball; nobody else follows it.
                if obs.can_kick(player.id):
                    # Strike it, from wherever we are. Once kick_aim puts the
                    # ball where it was aimed, a full-power shot from distance
                    # is worth taking rather than a hoof to nobody.
                    actions[player.id] = {
                        "movement": (0.0, 0.0),
                        "kick_direction": kick_aim(obs, self.aim(obs, player), 1.0),
                        "kick_power": 1.0}
                else:
                    # Run to where the ball is going, not where it is.
                    self.run_to(actions, obs, player, meet, blocked)

            else:
                # Everyone else spreads out ahead of the ball, one lane each.
                # Two lines, and no cleverness at all: the lane is fixed to the
                # slot, so these four never swap sides however the play moves.
                #
                # Centred on the pitch, which is the part that took fixing.
                # Numbering the lanes off slot 2 put them at -12, 0, +12 and
                # +24 on a pitch that runs from -30 to +30: the whole shape
                # sat six units into one channel, the widest slot spent the
                # match pinned against a touchline, and a sixth player would
                # have been pushed off the pitch entirely.
                outfield = len(obs.my_players) - 1
                lane = ((player.id - (outfield + 1) * 0.5)
                        * (obs.field.height * 0.2))
                spot = (lead[0] + SHAPE_DEPTH, lane)
                self.run_to(actions, obs, player, spot, blocked)

        return actions

    def off_the_ball(self, obs):
        actions = {}
        path = BallPath(obs)
        chaser_id, meet = self.chase(obs, path)
        blocked = their_restart(obs)

        # Whoever is next-nearest the ball supports the chaser instead of
        # picking up a man. One player at the ball wins the race and then
        # loses every loose ball that comes off it, and a loose ball in their
        # half is the cheapest shot on offer. It also stops the marking line
        # drifting into their kickoff circle, which is where the fouls were
        # coming from.
        bx, by = obs.ball.position
        rest = [p for p in obs.my_players if p.id != 0 and p.id != chaser_id]
        second_id = None
        if len(rest) > 1:
            second = min(rest, key=lambda p: (p.position[0] - bx) ** 2
                         + (p.position[1] - by) ** 2)
            second_id = second.id
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
                gx, gy = obs.my_goal
                dx, dy = bx - gx, by - gy
                gap = math.hypot(dx, dy)
                if gap > 1e-6:
                    spot = (gx + dx / gap * COVER_DEPTH,
                            gy + dy / gap * COVER_DEPTH)
                else:
                    spot = (gx + COVER_DEPTH, gy)
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
