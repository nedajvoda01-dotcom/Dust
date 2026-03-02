"""src/audio — Stage 36 Generative Foley from Physics.

All sound is synthesised from contact physics; no wav/ogg assets are used.

Modules
-------
ContactImpulseCollector  — aggregate ContactImpulse events from the physics layer
MaterialAcousticDB       — per-material acoustic profiles (modal params)
ExcitationGenerator      — turn a ContactImpulse into an excitation signal
ModalResonator           — modal-synthesis resonator (budget-capped pool)
SpatialEmitter           — inverse-square attenuation + distance low-pass
AtmosphericPropagation   — dust low-pass + cave delay-network reverb
MegaResonator            — bulk plate resonator for rifts / avalanches
"""
