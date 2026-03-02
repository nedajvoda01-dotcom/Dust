"""MaterialToAudioAdapter — Stage 45 material state → acoustic profile.

Blends :class:`~src.audio.MaterialAcousticDB.MaterialAcousticProfile`
according to the active surface material state fields:

* dustThickness ↑  → more granular noise (mix toward MAT_DUST)
* crustHardness ↑  → brittle click profile (mix toward MAT_FRACT)
* snowCompaction ↑ → crunch/creak (mix toward MAT_SNOW or MAT_ICE)
* iceFilm ↑        → stick-slip squeal (mix toward MAT_ICE)

Also converts :class:`~src.material.PhaseChangeSystem.BrittleEvent`
objects into :class:`~src.audio.ContactImpulseCollector.ContactImpulse`
records for injection into the audio pipeline.

Public API
----------
MaterialToAudioAdapter(db=None)
  .profile_for(state, base_mat_id=MAT_BASALT)
      -> MaterialAcousticProfile
  .brittle_impulses(event)
      -> List[ContactImpulse]
"""
from __future__ import annotations

from typing import List, Optional

from src.audio.MaterialAcousticDB import (
    MaterialAcousticDB,
    MaterialAcousticProfile,
    MAT_BASALT,
    MAT_DUST,
    MAT_FRACT,
    MAT_ICE,
    MAT_SNOW,
)
from src.audio.ContactImpulseCollector import ContactImpulse
from src.material.SurfaceMaterialState import SurfaceMaterialState
from src.material.PhaseChangeSystem import BrittleEvent


class MaterialToAudioAdapter:
    """Maps surface material state to acoustic profiles and impulses.

    Parameters
    ----------
    db : MaterialAcousticDB or None
        Acoustic profile database.  If None, a default instance is created.
    """

    def __init__(self, db: Optional[MaterialAcousticDB] = None) -> None:
        self._db = db if db is not None else MaterialAcousticDB()

    def profile_for(
        self,
        state: SurfaceMaterialState,
        base_mat_id: int = MAT_BASALT,
    ) -> MaterialAcousticProfile:
        """Compute a blended acoustic profile for the given state.

        Blending priority (applied sequentially):
        1. Start from ``base_mat_id`` profile.
        2. Mix toward DUST proportional to ``dustThickness``.
        3. Mix toward FRACT proportional to ``crustHardness``.
        4. Mix toward SNOW proportional to ``snowCompaction``.
        5. Mix toward ICE proportional to ``iceFilm``.

        Parameters
        ----------
        state :
            Current surface material state.
        base_mat_id :
            Starting material ID (e.g. MAT_BASALT, MAT_DEBRIS).
        """
        profile = self._db.get(base_mat_id)

        if state.dust_thickness > 0.0:
            dust_p = self._db.get(MAT_DUST)
            profile = MaterialAcousticDB.mix(profile, dust_p,
                                             state.dust_thickness * 0.7)

        if state.crust_hardness > 0.0:
            fract_p = self._db.get(MAT_FRACT)
            profile = MaterialAcousticDB.mix(profile, fract_p,
                                             state.crust_hardness * 0.5)

        if state.snow_compaction > 0.0:
            snow_p = self._db.get(MAT_SNOW)
            profile = MaterialAcousticDB.mix(profile, snow_p,
                                             state.snow_compaction * 0.6)

        if state.ice_film > 0.0:
            ice_p = self._db.get(MAT_ICE)
            profile = MaterialAcousticDB.mix(profile, ice_p,
                                             state.ice_film * 0.8)

        return profile

    def brittle_impulses(self, event: BrittleEvent) -> List[ContactImpulse]:
        """Convert a :class:`BrittleEvent` into micro-impulses.

        Generates ``event.impulse_count`` ContactImpulse objects with
        short duration and high slip ratio (brittle crack character).
        """
        impulses: List[ContactImpulse] = []
        base_magnitude = event.hardness_before * 0.8
        for i in range(event.impulse_count):
            # Stagger magnitudes slightly for a natural crackle
            mag = base_magnitude * (1.0 - i * 0.15)
            impulses.append(ContactImpulse(
                impulse_magnitude=max(0.05, mag),
                contact_duration=0.002 + i * 0.001,
                material_pair=(MAT_FRACT, MAT_FRACT),
                slip_ratio=0.8,
                contact_area=0.001,
                world_pos=(float(event.ix), 0.0, float(event.iy)),
            ))
        return impulses
