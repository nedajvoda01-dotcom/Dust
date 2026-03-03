"""observer — Stage 53 Observer Influence Without Agency.

The player is a physical body in the world.  Their mass, heat, footsteps,
impulse events, and micro-vibrations contribute minimally but deterministically
to the planet's existing fields.  This is NOT gameplay influence — there are no
bonuses, triggers, or events specific to the player.  The player is simply
one of many sources in the system.

Modules
-------
InfluenceLimiter        — caps per-tile and global player contributions.
ContactStressInjector   — footstep/fall contact forces → stress fields.
ThermalFootprintInjector— body heat → local ice/snow phase shift.
ImpulseToShearInjector  — jump/grasp impulses → shear stress.
PlayerInfluenceAdapter  — main orchestrator that calls all injectors.
"""
