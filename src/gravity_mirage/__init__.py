import numpy as np
"""
Gravity Mirage: Simulador de lensing gravitacional
"""
from .physics import SchwarzschildBlackHole
from .ray_tracer import GravitationalRayTracer

def main() -> None:
    """Función principal del simulador"""
    print(" Gravity Mirage - Simulador de Agujero Negro")
    print("=" * 50)
    
    # Crear un agujero negro de 10 masas solares
    bh = SchwarzschildBlackHole(mass=10)
    print(f"Radio de Schwarzschild: {bh.schwarzschild_radius:.2f} m")
    print(f"                       : {bh.schwarzschild_radius/1000:.2f} km")
    
    # Crear ray tracer
    tracer = GravitationalRayTracer(bh)
    
    # Ejemplo simple de deflexión
    impact_param = 100 * bh.schwarzschild_radius
    angle = bh.deflection_angle_weak_field(impact_param)
    print(f"\nÁngulo de deflexión a {impact_param/bh.schwarzschild_radius:.1f} Rs:")
    print(f"  {np.degrees(angle):.6f} grados")
    print(f"  {angle * 3600:.3f} arcosegundos")
    

__all__ = ['SchwarzschildBlackHole', 'GravitationalRayTracer', 'main']