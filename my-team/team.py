import math
import random

from soccer import (
    TeamAction, TeamController, clamp, closest_to_ball, direction, distance,
    normalise,
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

# How far off his line the keeper stands.
KEEPER_DEPTH = 2.0


def ball_path(obs, ticks=HORIZON):
    """The ball's predicted position every SCAN_STEP ticks, for `ticks` ticks.

    Same maths the engine applies to the ball, so with nobody in the way this
    agrees with the simulator exactly. Built once per tick and shared by every
    player, rather than re-rolled for each of them.

    The roll still runs a tick at a time, because a wall bounce has to be
    caught on the tick it happens. Only the ticks `intercept` actually looks
    at are kept, though: it reads one point in every SCAN_STEP, so storing all
    of them built a hundred and twenty tuples a tick to throw away sixty.
    """
    f = obs.field
    dt = 1.0 / f.simulation_hz
    x, y = obs.ball.position
    vx, vy = obs.ball.velocity
    # The ball centre stops one radius short of the boundary.
    hx = f.width / 2 - f.ball_radius
    hy = f.height / 2 - f.ball_radius

    friction = f.ball_friction
    path = [(x, y)]
    for _ in range(ticks // SCAN_STEP):
        for _ in range(SCAN_STEP):
            x, y = x + vx * dt, y + vy * dt
            vx, vy = vx * friction, vy * friction
            # Reflect back across whichever wall it went through.
            if y > hy:
                y, vy = 2 * hy - y, -vy * WALL_RESTITUTION
            elif y < -hy:
                y, vy = -2 * hy - y, -vy * WALL_RESTITUTION
            if x > hx:
                x, vx = 2 * hx - x, -vx * WALL_RESTITUTION
            elif x < -hx:
                x, vx = -2 * hx - x, -vx * WALL_RESTITUTION
        path.append((x, y))
    return path


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


def intercept(player, path, reach):
    """Earliest point on the ball's path this player can actually get to.

    Walks the prediction forward and takes the first moment where the player
    can already be there. That point, not the ball's current position, is
    where they should be running.

    Compares squared distances against `reach` rather than working out an
    arrival time per point: same answer, no square roots, and cheap enough to
    run for every player on every tick.
    """
    px, py = player.position
    for i, (x, y) in enumerate(path):
        dx, dy = px - x, py - y
        r = reach[i]
        if dx * dx + dy * dy <= r * r:
            return i, (x, y)
    # Out of reach inside the horizon: head for the end of the path anyway.
    return len(path), path[-1]


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
    version = "8"

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
        actions.move(player.id, direction(player.position, spot))

    def meeting_points(self, obs, path):
        """Where each outfield player would meet the ball, and how soon."""
        # The table depends only on the physics, and those do not change while
        # a match is running, so it is built on the first tick and reused for
        # the rest. Rebuilding it every tick was about a fifth of the whole
        # decision, and the decision time is what separates teams on equal
        # points.
        reach = self._reach
        if reach is None or len(reach) < len(path):
            reach = self._reach = reach_table(obs, (len(path) - 1) * SCAN_STEP)
        meets = {
            player.id: intercept(player, path, reach)
            for player in obs.my_players
            if player.id != 0
        }
        # A short-handed side would otherwise leave this empty.
        if not meets:
            meets = {p.id: intercept(p, path, reach) for p in obs.my_players}
        return meets

    def keep_goal(self, actions, obs, keeper, lead):
        """Hold the line on the ball's predicted y, and clear anything close.

        Standing in the way is not enough: the ball keeps most of its speed
        off a body, so a shot the keeper merely blocks carries on into the
        net. Kicking it is what turns a block into a save.
        """
        if obs.can_kick(keeper.id):
            # Clear it upfield, hard, and away from the middle.
            aim = (obs.opponent_goal[0], obs.ball.position[1] * 3.0)
            actions.kick(keeper.id, kick_aim(obs, aim, 1.0), kick_power=1.0)
            return
        # Track the ball's y, and stay deep. Coming out to narrow the angle
        # only looks right: a block is not a save here, because the ball keeps
        # most of its speed off a body. The clearance above is the save, and
        # you have to be on the ball to make it.
        mouth = obs.field.goal_width / 2
        spot = (obs.my_goal[0] + KEEPER_DEPTH, clamp(lead[1], -mouth, mouth))
        actions.move(keeper.id, direction(keeper.position, spot))

    def on_the_ball(self, obs):
        actions = TeamAction()
        path = ball_path(obs)
        meets = self.meeting_points(obs, path)
        # Whoever gets there soonest, which is not always whoever is nearest.
        chaser_id = min(meets, key=lambda pid: meets[pid][0])
        # Where the ball will be while the shape is being taken up.
        lead = path[min(SHAPE_LEAD // SCAN_STEP, len(path) - 1)]
        blocked = their_restart(obs)

        # Every one of your players goes through this loop exactly once and
        # leaves it with exactly one action.
        for player in obs.my_players:
            if player.id == 0:
                # Keeper: hold the goal line, slide across with the ball.
                self.keep_goal(actions, obs, player, lead)

            elif player.id == chaser_id:
                # The one player meeting the ball; nobody else follows it.
                if obs.can_kick(player.id):
                    # Strike it, from wherever we are. Once kick_aim puts the
                    # ball where it was aimed, a full-power shot from distance
                    # is worth taking rather than a hoof to nobody.
                    actions.kick(player.id,
                                 kick_aim(obs, self.aim(obs, player), 1.0),
                                 kick_power=1.0)
                else:
                    # Run to where the ball is going, not where it is.
                    self.run_to(actions, obs, player,
                                meets[player.id][1], blocked)

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
        actions = TeamAction()
        path = ball_path(obs)
        meets = self.meeting_points(obs, path)
        chaser_id = min(meets, key=lambda pid: meets[pid][0])
        lead = path[min(SHAPE_LEAD // SCAN_STEP, len(path) - 1)]
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
        pairs = self.marking(obs, rest)

        for player in obs.my_players:
            if player.id == 0:
                self.keep_goal(actions, obs, player, lead)

            elif player.id == chaser_id:
                # Cut the ball off rather than following it around.
                self.run_to(actions, obs, player,
                            meets[player.id][1], blocked)

            elif player.id == second_id:
                # Second man. He stands off the ball rather than on top of the
                # chaser, and goalside of it, so a ball that squirts loose runs
                # to him and not to them.
                self.run_to(actions, obs, player, (bx - 6.0, by), blocked)

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
