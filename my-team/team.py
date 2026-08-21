"""Your team. Edit this file.

Everything you write goes in here. A submission is one file of code — extra
.py files next to it are not importable when the server loads your team, so
they are rejected rather than silently ignored.

This is a working team, so you can play it right now and see numbers come back:

    START.cmd play my-team --against balanced      (Windows)
    ./start.sh play my-team --against balanced     (macOS and Linux)

It is about as simple as a team can be while still playing football. Every
player asks two questions and does one of four things:

    is the ball ours?
        am I the closest of us to it?   ->  shoot at their goal
        otherwise                       ->  push up towards their goal
    otherwise
        am I the closest of us to it?   ->  go and get it
        otherwise                       ->  drop back towards our goal

It will lose to the reference opponents, which is the point — it is something
to improve, not something to hand in as it stands. Things wrong with it, in
roughly the order they cost you goals: nobody keeps goal, the four players who
are not chasing all run to the same place, nobody ever passes, and every shot
is struck at full power from wherever the player is standing.

The other half of your submission is `team.toml` next to this file, which says
who you are. Fill that in too.

You write one method, `act`, and it is called on every tick of the match —
whether the ball is yours or theirs. Deciding which of those it is, and what
your players should be doing about it, is the assignment. There is no engine
call that will do it for you: an observation reports where everybody is and how
the ball behaves, and the football is yours to write.

You are always given the same pitch whichever side you are really playing:
your goal at the left, theirs at the right, forward is +x, your players are
ids 0..n-1. Write one team, not one per side.

The full API is in the student guide that came with this download.
"""

"""my-team/team.py - a complete, working first team."""

from soccer import (
    TeamAction, TeamController, clamp, closest_to_ball, direction, distance,
)


class MyTeam(TeamController):
    name = "Connor Pace"
    version = "1"

    def act(self, obs):
        # The one method the engine calls. Everything starts with deciding
        # what kind of moment this is; this team asks the simplest question
        # there is, and you should expect to outgrow it.
        if closest_to_ball(obs):
            return self.on_the_ball(obs)
        return self.off_the_ball(obs)

    def on_the_ball(self, obs):
        actions = TeamAction()
        chaser = obs.closest_my_player_to(obs.ball.position)

        # Every one of your players goes through this loop exactly once and
        # leaves it with exactly one action.
        for player in obs.my_players:
            if player.id == 0:
                # Keeper: hold the goal line, slide across with the ball.
                mouth = obs.field.goal_width / 2
                spot = (obs.my_goal[0] + 2.0,
                        clamp(obs.ball.position[1], -mouth, mouth))
                actions.move(player.id, direction(player.position, spot))

            elif player.id == chaser.id:
                # The nearest player, and only that one, goes to the ball.
                if obs.can_kick(player.id):
                    # Power sized to the distance: see "How far a kick goes".
                    gap = distance(player.position, obs.opponent_goal)
                    actions.kick(
                        player.id,
                        direction(player.position, obs.opponent_goal),
                        kick_power=min(1.0, gap / 33.0),
                    )
                else:
                    actions.move(
                        player.id,
                        direction(player.position, obs.ball.position),
                    )

            else:
                # Everyone else spreads out ahead of the ball, one lane each.
                # Two lines, and no cleverness at all: the lane is fixed to the
                # slot, so these four never swap sides however the play moves.
                lane = (player.id - 2) * (obs.field.height * 0.2)
                spot = (obs.ball.position[0] + 15.0, lane)
                actions.move(player.id, direction(player.position, spot))

        return actions

    def off_the_ball(self, obs):
        actions = TeamAction()
        chaser = obs.closest_my_player_to(obs.ball.position)

        for player in obs.my_players:
            if player.id == 0:
                mouth = obs.field.goal_width / 2
                spot = (obs.my_goal[0] + 2.0,
                        clamp(obs.ball.position[1], -mouth, mouth))
                actions.move(player.id, direction(player.position, spot))

            elif player.id == chaser.id:
                actions.move(
                    player.id, direction(player.position, obs.ball.position)
                )

            else:
                # Mark goalside: stand between the nearest opponent and the
                # goal you defend, which is always at -x. Nothing stops two of
                # your players picking the same opponent — see section 11.
                them = obs.closest_opponent_to(player.position)
                if them is None:
                    spot = obs.my_goal
                else:
                    spot = (them.position[0] - 3.0, them.position[1])
                actions.move(player.id, direction(player.position, spot))

        return actions
