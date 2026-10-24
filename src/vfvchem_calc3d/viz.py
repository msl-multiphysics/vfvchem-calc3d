"""PyVista rendering of obj groups, domain surface triangulations, and control volumes."""

from __future__ import annotations

import numpy as np

from .geometry import Dual3D


def _region_colors(num: int, cmap_name: str) -> list[tuple]:
    """RGB colors from a categorical colormap."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    return [cmap(i % cmap.N)[:3] for i in range(num)]


def _contrast_text_color(rgb) -> str:
    r, g, b = rgb[:3]
    # Rec. 601 luma; light backgrounds get black text
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


def _region_surface(polys: list[np.ndarray]):
    """Polygonal surface of a group."""
    import pyvista as pv

    pts = np.concatenate(polys)
    cells = []
    n = 0
    for poly in polys:
        cells.extend([len(poly), *range(n, n + len(poly))])
        n += len(poly)
    return pv.PolyData(pts, faces=cells)


def _segments_to_lines(segs: np.ndarray):
    """(n_seg, 2, 3) segments -> PyVista line PolyData."""
    import pyvista as pv

    n = len(segs)
    lines = np.column_stack([np.full(n, 2), 2 * np.arange(n), 2 * np.arange(n) + 1]).ravel()
    return pv.PolyData(segs.reshape(-1, 3), lines=lines)


def _label_point(surf) -> np.ndarray:
    """Face center nearest the area-weighted centroid, so the label stays on curved surfaces."""
    tri = surf.triangulate()
    centers = tri.cell_centers().points
    areas = tri.compute_cell_sizes(length=False, volume=False)["Area"]
    centroid = (centers * areas[:, None]).sum(axis=0) / areas.sum()
    return centers[[np.argmin(np.linalg.norm(centers - centroid, axis=1))]]


def _text_button_texture(label: str, font_path: str | None = None, font_size: int = 16):
    """Build a PyVista ImageData texture with a labeled button appearance."""
    import pyvista as pv
    from PIL import Image, ImageDraw, ImageFont

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x, pad_y = 14, 8
    width = text_w + 2 * pad_x
    height = text_h + 2 * pad_y

    img = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(60, 60, 60), width=2)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), label, fill=(20, 20, 20), font=font)

    arr = np.asarray(img, dtype=np.uint8)  # (height, width, 3), y=0 at top
    # VTK ImageData is x-fastest with y=0 at bottom
    arr = np.flipud(arr)
    button = pv.ImageData(dimensions=(width, height, 1))
    button.point_data["texture"] = arr.reshape(width * height, 3)
    return button, width, height


def _add_text_button(plotter, label: str, position: tuple[float, float], on_click, *, font_path: str | None = None):
    """Add a clickable text button at the given bottom-left pixel position."""
    from vtkmodules.vtkCommonCore import vtkCommand
    from vtkmodules.vtkInteractionWidgets import (
        vtkButtonWidget,
        vtkTexturedButtonRepresentation2D,
    )

    texture, width, height = _text_button_texture(label, font_path=font_path)

    # Same texture for both states so the control does not look like a toggle.
    # Do not force SetState(0) in the callback; that re-enters StateChangedEvent
    # and can swallow camera updates on some VTK/PyVista builds.
    button_rep = vtkTexturedButtonRepresentation2D()
    button_rep.SetNumberOfStates(2)
    button_rep.SetButtonTexture(0, texture)
    button_rep.SetButtonTexture(1, texture)
    button_rep.SetState(0)
    button_rep.SetPlaceFactor(1)
    button_rep.PlaceWidget(
        [position[0], position[0] + width, position[1], position[1] + height, 0.0, 0.0]
    )

    button_widget = vtkButtonWidget()
    button_widget.SetInteractor(plotter.iren.interactor)
    button_widget.SetRepresentation(button_rep)
    button_widget.SetCurrentRenderer(plotter.renderer)
    button_widget.On()

    def _on_click(widget, event):
        on_click(widget, event)

    button_widget.AddObserver(vtkCommand.StateChangedEvent, _on_click)
    plotter.widgets.button_widgets.append(button_widget)
    return button_widget, width, height


def _add_viewer_buttons(
    plotter,
    *,
    category_actors: dict[str, list],
    initial_camera: dict,
    margin: float = 10.0,
    gap: float = 6.0,
):
    """Stack View Domains / Boundaries / Interfaces above Reset Camera (bottom-right).

    View buttons toggle the shading and labels.
    ``initial_camera`` is a dict with key ``\"cpos\"`` filled on the first real render.
    """
    font_path = _resolve_label_font("IBM Plex Sans")
    win_w, _win_h = plotter.window_size

    # Top to bottom in the stack; Reset Camera is last (lowest)
    stack = [
        ("View Domains", "domains"),
        ("View Boundaries", "boundaries"),
        ("View Interfaces", "interfaces"),
        ("Reset Camera", None),
    ]

    def _apply_view(mode: str, *, render: bool = True):
        for key, actors in category_actors.items():
            for actor in actors:
                actor.SetVisibility(mode == key)
        if render:
            plotter.render()

    # Start with boundaries only. Skip render so we do not capture a pre-show
    # camera; plotter.show() performs the first real render.
    _apply_view("boundaries", render=False)

    # Place from bottom upward so Reset Camera sits at the bottom of the stack
    y = margin
    for label, key in reversed(stack):
        _tex, w, h = _text_button_texture(label, font_path=font_path)
        position = (win_w - w - margin, y)

        if key is None:
            def _on_reset(_widget, _event, _cam=initial_camera):
                cpos = _cam.get("cpos")
                if cpos is None:
                    return
                plotter.camera_position = cpos
                plotter.render()

            _add_text_button(plotter, label, position, _on_reset, font_path=font_path)
        else:
            def _on_view(_widget, _event, _key=key):
                _apply_view(_key)

            _add_text_button(plotter, label, position, _on_view, font_path=font_path)

        y += h + gap


def _resolve_label_font(family: str = "IBM Plex Sans") -> str | None:
    """Return a TTF path for the label font, or None if unavailable."""
    try:
        from matplotlib import font_manager
    except ImportError:
        return None
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except (ValueError, RuntimeError):
        return None
    if path and "DejaVu" not in path:  # matplotlib's silent fallback name
        return path
    # Explicit scan in case findfont fell back quietly on some versions
    for entry in font_manager.fontManager.ttflist:
        if entry.name == family and entry.style == "normal" and int(entry.weight) == 400:
            return entry.fname
    for entry in font_manager.fontManager.ttflist:
        if entry.name == family and entry.style == "normal":
            return entry.fname
    return None


def render_geometry(
    grp2d: dict[str, list[np.ndarray]],
    grp3d: dict[str, list[np.ndarray]],
    duals: dict[str, Dual3D],
    *,
    show_cvol: bool = False,
    show_labels: bool = True,
):
    """Open an interactive PyVista window with dom/bnd/itf groups and their names.

    Loaded domains show their surface triangulation; show_cvol adds the control volume outlines on the surface.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "show_mesh() requires pyvista. Install with: pip install pyvista"
        ) from exc

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.show_axes()

    # face outlines stay visible in every view; edges inside a face are hidden
    outlines = [
        _region_surface(polys).clean().extract_feature_edges(
            boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
        )
        for polys in grp2d.values()
    ]
    outline = pv.merge(outlines).clean() if outlines else pv.PolyData()
    if outline.n_points > 0:
        plotter.add_mesh(outline, color=(0.25, 0.25, 0.25), line_width=2)

    # surface triangulation and control volume outlines of loaded domains
    for name in sorted(duals):
        dual = duals[name]
        n = len(dual.bnd_tri)
        faces = np.column_stack([np.full(n, 3), dual.bnd_tri]).ravel()
        plotter.add_mesh(pv.PolyData(dual.pos, faces=faces), style="wireframe", color=(0.35, 0.35, 0.35), line_width=1)
        if show_cvol:
            # segments from each edge midpoint to the circumcenter of the surface triangle
            segs = []
            for r in range(3):
                mid = 0.5 * (dual.pos[dual.bnd_tri[:, r]] + dual.pos[dual.bnd_tri[:, (r + 1) % 3]])
                segs.append(np.stack([mid, dual.bnd_circ], axis=1))
            plotter.add_mesh(_segments_to_lines(np.concatenate(segs)), color="dimgray", line_width=1)

    # translucent category shading and labels (toggled by View buttons)
    category_actors: dict[str, list] = {"domains": [], "boundaries": [], "interfaces": []}
    label_items = []  # (category, label point, name, color)
    for category, groups, cmap_name in (
        ("domains", grp3d, "Pastel1"),
        ("boundaries", {g: p for g, p in grp2d.items() if g.startswith("bnd")}, "Set1"),
        ("interfaces", {g: p for g, p in grp2d.items() if g.startswith("itf")}, "Set2"),
    ):
        names = sorted(groups)
        for name, color in zip(names, _region_colors(len(names), cmap_name)):
            surf = _region_surface(groups[name])
            actor = plotter.add_mesh(surf.triangulate(), color=color, opacity=0.35)
            category_actors[category].append(actor)
            label_items.append((category, _label_point(surf), name, color))

    if show_labels:
        font_file = _resolve_label_font("IBM Plex Sans")
        for category, point, name, rgb in label_items:
            actor = plotter.add_point_labels(
                point,
                [name],
                point_size=5,
                font_size=14,
                bold=False,
                font_file=font_file,
                text_color=_contrast_text_color(rgb),
                shape_color=rgb,
                shape_opacity=0.8,
                always_visible=True,
            )
            category_actors[category].append(actor)

    # x right, z up, y into the page (camera on -y looking toward +y)
    plotter.view_xz()

    # Capture the true startup camera on the first render inside show().
    # Saving before show() is wrong: show() re-frames the camera afterward.
    initial_camera: dict = {"cpos": None}

    def _capture_initial_camera(pl):
        if initial_camera["cpos"] is None:
            initial_camera["cpos"] = pl.camera_position

    plotter.add_on_render_callback(_capture_initial_camera)

    _add_viewer_buttons(
        plotter,
        category_actors=category_actors,
        initial_camera=initial_camera,
    )

    plotter.show()
