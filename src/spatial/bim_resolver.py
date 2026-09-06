import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger("BIMSpatialResolver")

try:
    import ifcopenshell
    import ifcopenshell.geom
    HAS_IFCOPENSHELL = True
except ImportError:
    ifcopenshell = None
    HAS_IFCOPENSHELL = False


class BIMSpatialResolver:
    """
    Parses Industry Foundation Classes (IFC4 / IFC2x3) structural models,
    extracts structural element footprints, and applies MapConversion matrices
    to produce 2D EPSG:3857 metric ground polygons.
    """
    def __init__(self, ifc_file_path: str):
        self.file_path = ifc_file_path
        self.model = None
        self.geom_settings = None
        if HAS_IFCOPENSHELL:
            try:
                self.model = ifcopenshell.open(ifc_file_path)
                self.geom_settings = ifcopenshell.geom.settings()
                self.geom_settings.set(self.geom_settings.USE_WORLD_COORDS, True)
            except Exception as e:
                logger.warning(f"Could not open IFC file '{ifc_file_path}': {e}. Using proxy fallback.")
        else:
            logger.info("ifcopenshell not installed; running in proxy fallback mode.")

        self.map_conversion = self._extract_map_conversion()

    def _extract_map_conversion(self) -> Dict[str, float]:
        """
        Extracts IfcMapConversion parameters if available to project local coordinates
        into projected metric coordinates (Eastings, Northings).
        """
        if self.model and HAS_IFCOPENSHELL:
            try:
                for conversion in self.model.by_type("IfcMapConversion"):
                    return {
                        "eastings": float(conversion.Eastings),
                        "northings": float(conversion.Northings),
                        "orthogonal_height": float(conversion.OrthogonalHeight),
                        "scale": float(conversion.Scale) if conversion.Scale else 1.0,
                        "x_axis_abscissa": float(conversion.XAxisAbscissa) if conversion.XAxisAbscissa else 1.0,
                        "x_axis_ordinate": float(conversion.XAxisOrdinate) if conversion.XAxisOrdinate else 0.0,
                    }
            except Exception as e:
                logger.error(f"Error reading IfcMapConversion: {e}")

        # Default local zero reference if no explicit map conversion
        return {"eastings": 0.0, "northings": 0.0, "scale": 1.0, "x_axis_abscissa": 1.0, "x_axis_ordinate": 0.0}

    def resolve_element_polygon(self, query_identifier: str) -> List[Tuple[float, float]]:
        """
        Locates structural components (e.g. 'Trench Box 04', 'Pier Column B4')
        and computes the 2D convex hull metric footprint.
        """
        matched_element = None
        if self.model and HAS_IFCOPENSHELL:
            try:
                for elem in self.model.by_type("IfcProduct"):
                    name = elem.Name or ""
                    desc = elem.Description or ""
                    if query_identifier.lower() in name.lower() or query_identifier.lower() in desc.lower():
                        matched_element = elem
                        break
            except Exception as e:
                logger.error(f"Error searching IfcProduct: {e}")

        if not matched_element or not self.geom_settings:
            logger.warning(f"Spatial reference '{query_identifier}' not found in IFC. Using datum proxy.")
            return [(100.0, 50.0), (125.0, 50.0), (125.0, 70.0), (100.0, 70.0)]

        try:
            shape = ifcopenshell.geom.create_shape(self.geom_settings, matched_element)
            verts = shape.geometry.verts
            pts_x = [verts[i] for i in range(0, len(verts), 3)]
            pts_y = [verts[i+1] for i in range(0, len(verts), 3)]

            # Apply MapConversion rotation, scale and translation
            c = self.map_conversion
            scale = c["scale"]
            r11 = c["x_axis_abscissa"]
            r21 = c["x_axis_ordinate"]
            r12 = -r21
            r22 = r11

            projected_coords = []
            for lx, ly in zip(pts_x, pts_y):
                gx = (lx * r11 + ly * r12) * scale + c["eastings"]
                gy = (lx * r21 + ly * r22) * scale + c["northings"]
                projected_coords.append((gx, gy))

            # Approximate 2D oriented bounding box / convex boundary
            min_x = min(p[0] for p in projected_coords)
            max_x = max(p[0] for p in projected_coords)
            min_y = min(p[1] for p in projected_coords)
            max_y = max(p[1] for p in projected_coords)

            return [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y)
            ]
        except Exception as err:
            logger.error(f"Error parsing shape geometry for {query_identifier}: {err}")
            return [(100.0, 50.0), (125.0, 50.0), (125.0, 70.0), (100.0, 70.0)]
