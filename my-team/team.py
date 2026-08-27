from soccer import (
    TeamAction, TeamController, clamp, closest_to_ball, direction, distance,
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

# Only test every Nth tick of the path for a meeting point. Two ticks is a
# tenth of a second, which is far finer than the difference matters, and it
# keeps a long horizon cheap enough to run every tick.
SCAN_STEP = 2


def ball_path(obs, ticks=HORIZON):
    """The ball's predicted position at each of the next `ticks` ticks.

    Same maths the engine applies to the ball, so with nobody in the way this
    agrees with the simulator exactly. Built once per tick and shared by every
    player, rather than re-rolled for each of them.
    """
    f = obs.field
    dt = 1.0 / f.simulation_hz
    x, y = obs.ball.position
    vx, vy = obs.ball.velocity
    # The ball centre stops one radius short of the boundary.
    hx = f.width / 2 - f.ball_radius
    hy = f.height / 2 - f.ball_radius

    path = [(x, y)]
    for _ in range(ticks):
        x, y = x + vx * dt, y + vy * dt
        vx, vy = vx * f.ball_friction, vy * f.ball_friction
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
    """How far a player can run in 0, 1, 2 ... `ticks` ticks.

    Players accelerate from a standstill, so distance / max_speed alone is
    optimistic over short runs; this adds the ramp up to top speed. It depends
    only on the physics, so it is built once a tick and shared by everyone.
    """
    f = obs.field
    ramp = f.max_speed / f.acceleration            # seconds spent accelerating
    ramp_gap = 0.5 * f.acceleration * ramp * ramp  # ground covered doing it
    # Being within kick range counts as having arrived.
    slack = f.kick_range * 0.5

    table = []
    for t in range(ticks + 1):
        seconds = t / f.simulation_hz
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
    for t in range(0, len(path), SCAN_STEP):
        x, y = path[t]
        dx, dy = px - x, py - y
        r = reach[t]
        if dx * dx + dy * dy <= r * r:
            return t, path[t]
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
    edge = obs.field.centre_circle_radius + obs.field.player_radius
    gap = distance(spot, (0.0, 0.0))
    if gap >= edge:
        return spot
    if gap < 1e-6:
        return (-edge, 0.0)      # dead centre: back off into our own half
    return (spot[0] * edge / gap, spot[1] * edge / gap)


class MyTeam(TeamController):
    name = "Connor Pace"
    version = "2"

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
        reach = reach_table(obs, len(path) - 1)
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
            actions.kick(keeper.id, direction(keeper.position, aim),
                         kick_power=1.0)
            return
        mouth = obs.field.goal_width / 2
        spot = (obs.my_goal[0] + 2.0, clamp(lead[1], -mouth, mouth))
        actions.move(keeper.id, direction(keeper.position, spot))

    def on_the_ball(self, obs):
        actions = TeamAction()
        path = ball_path(obs)
        meets = self.meeting_points(obs, path)
        # Whoever gets there soonest, which is not always whoever is nearest.
        chaser_id = min(meets, key=lambda pid: meets[pid][0])
        # Where the ball will be while the shape is being taken up.
        lead = path[min(SHAPE_LEAD, len(path) - 1)]
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
                    # Power sized to the distance: see "How far a kick goes".
                    gap = distance(player.position, obs.opponent_goal)
                    actions.kick(
                        player.id,
                        direction(player.position, obs.opponent_goal),
                        kick_power=min(1.0, gap / 33.0),
                    )
                else:
                    # Run to where the ball is going, not where it is.
                    self.run_to(actions, obs, player,
                                meets[player.id][1], blocked)

            else:
                # Everyone else spreads out ahead of the ball, one lane each.
                # Two lines, and no cleverness at all: the lane is fixed to the
                # slot, so these four never swap sides however the play moves.
                lane = (player.id - 2) * (obs.field.height * 0.2)
                spot = (lead[0] + 15.0, lane)
                self.run_to(actions, obs, player, spot, blocked)

        return actions

    def off_the_ball(self, obs):
        actions = TeamAction()
        path = ball_path(obs)
        meets = self.meeting_points(obs, path)
        chaser_id = min(meets, key=lambda pid: meets[pid][0])
        lead = path[min(SHAPE_LEAD, len(path) - 1)]
        blocked = their_restart(obs)

        for player in obs.my_players:
            if player.id == 0:
                self.keep_goal(actions, obs, player, lead)

            elif player.id == chaser_id:
                # Cut the ball off rather than following it around.
                self.run_to(actions, obs, player,
                            meets[player.id][1], blocked)

            else:
                # Mark goalside: stand between the nearest opponent and the
                # goal you defend, which is always at -x. Nothing stops two of
                # your players picking the same opponent — see section 11.
                them = obs.closest_opponent_to(player.position)
                if them is None:
                    spot = obs.my_goal
                else:
                    spot = (them.position[0] - 3.0, them.position[1])
                self.run_to(actions, obs, player, spot, blocked)

        return actions
