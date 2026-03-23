#!/usr/bin/env python3
"""
Transform autolens workspace scripts from old plotting API to new API.
"""
import re
import sys
from pathlib import Path


def parse_output_from_mat_plot(text):
    m = re.search(r'aplt\.Output\s*\(path=([^,\)]+),\s*format=["\']([^"\']+)["\']', text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r'aplt\.Output\s*\(path=([^\)]+)\)', text)
    if m:
        return m.group(1).strip(), "png"
    return None, None


def extract_kwarg(text, kwarg_name):
    """Extract first value of kwarg_name=VALUE from text."""
    m = re.search(
        re.escape(kwarg_name) + r'\s*=\s*([\w\.]+(?:\([^)]*\))?)',
        text
    )
    return m.group(1).strip() if m else None


def build_output_args(output_path, output_format):
    if output_path is None:
        return ""
    return f", output_path={output_path}, output_format=\"{output_format or 'png'}\""


def collect_balanced(lines, start_idx):
    full = lines[start_idx]
    depth = full.count('(') - full.count(')')
    j = start_idx + 1
    while depth > 0 and j < len(lines):
        full += '\n' + lines[j]
        depth += lines[j].count('(') - lines[j].count(')')
        j += 1
    return full, j


# ─── Figures-2d mappings ───────────────────────────────────────────────────

FIGURES_2D = {
    'TracerPlotter': {
        'image':         lambda t, g: f'{t}.image_2d_from(grid={g})',
        'convergence':   lambda t, g: f'{t}.convergence_2d_from(grid={g})',
        'potential':     lambda t, g: f'{t}.potential_2d_from(grid={g})',
        'magnification': lambda t, g: f'{t}.magnification_2d_from(grid={g})',
    },
    'ImagingPlotter': {
        'data':                lambda d, _: f'{d}.data',
        'noise_map':           lambda d, _: f'{d}.noise_map',
        'psf':                 lambda d, _: f'{d}.psf',
        'signal_to_noise_map': lambda d, _: f'{d}.signal_to_noise_map',
    },
    'FitImagingPlotter': {
        'data':                    lambda f, _: f'{f}.data',
        'noise_map':               lambda f, _: f'{f}.noise_map',
        'model_image':             lambda f, _: f'{f}.model_data',
        'residual_map':            lambda f, _: f'{f}.residual_map',
        'normalized_residual_map': lambda f, _: f'{f}.normalized_residual_map',
        'chi_squared_map':         lambda f, _: f'{f}.chi_squared_map',
    },
    'InterferometerPlotter': {
        'data':        lambda d, _: f'{d}.data',
        'dirty_image': lambda d, _: f'{d}.dirty_image',
        'noise_map':   lambda d, _: f'{d}.noise_map',
    },
    'FitInterferometerPlotter': {
        'data':                    lambda f, _: f'{f}.data',
        'residual_map':            lambda f, _: f'{f}.residual_map',
        'normalized_residual_map': lambda f, _: f'{f}.normalized_residual_map',
    },
    'Array2DPlotter': {
        'array': lambda a, _: a,
    },
    # autogalaxy plotters
    'LightProfilePlotter': {
        'image': lambda lp, g: f'{lp}.image_2d_from(grid={g})',
    },
    'MassProfilePlotter': {
        'convergence':   lambda mp, g: f'{mp}.convergence_2d_from(grid={g})',
        'potential':     lambda mp, g: f'{mp}.potential_2d_from(grid={g})',
        'deflections_y': lambda mp, g: None,  # complex, needs separate lines
        'deflections_x': lambda mp, g: None,
    },
    'GalaxyPlotter': {
        'image': lambda gal, g: f'{gal}.image_2d_from(grid={g})',
    },
    'GalaxiesPlotter': {
        'image': lambda gals, g: f'{gals}.image_2d_from(grid={g})',
    },
    'InversionPlotter': {},
    'MapperPlotter': {},
    'PointDatasetPlotter': {},
}

SUBPLOT_MAP = {
    ('TracerPlotter', 'subplot_tracer'):
        lambda t, g, out: f"aplt.subplot_tracer(tracer={t}, grid={g}{out})",
    ('TracerPlotter', 'subplot_galaxies_images'):
        lambda t, g, out: f"aplt.subplot_galaxies_images(tracer={t}, grid={g}{out})",
    ('TracerPlotter', 'subplot_lensed_images'):
        lambda t, g, out: f"aplt.subplot_lensed_images(tracer={t}, grid={g}{out})",
    ('ImagingPlotter', 'subplot_dataset'):
        lambda d, _, out: f"aplt.subplot_imaging_dataset(dataset={d}{out})",
    ('FitImagingPlotter', 'subplot_fit'):
        lambda f, _, out: f"aplt.subplot_fit_imaging(fit={f}{out})",
    ('FitImagingPlotter', 'subplot_fit_log10'):
        lambda f, _, out: f"aplt.subplot_fit_imaging_log10(fit={f}{out})",
    ('FitImagingPlotter', 'subplot_of_planes'):
        lambda f, _, out: f"aplt.subplot_fit_imaging_of_planes(fit={f}{out})",
    ('FitImagingPlotter', 'subplot_tracer'):
        lambda f, _, out: f"aplt.subplot_fit_imaging_tracer(fit={f}{out})",
    ('FitInterferometerPlotter', 'subplot_fit'):
        lambda f, _, out: f"aplt.subplot_fit_interferometer(fit={f}{out})",
    ('FitInterferometerPlotter', 'subplot_real_space'):
        lambda f, _, out: f"aplt.subplot_fit_interferometer_real_space(fit={f}{out})",
    ('InterferometerPlotter', 'subplot_dataset'):
        lambda d, _, out: f"aplt.subplot_interferometer_dataset(dataset={d}{out})",
    ('InterferometerPlotter', 'subplot_dirty_images'):
        lambda d, _, out: f"aplt.subplot_interferometer_dirty_images(dataset={d}{out})",
    ('SubhaloPlotter', 'subplot_detection_imaging'):
        lambda r, extra, out: f"aplt.subplot_detection_imaging(result={r}, fit_imaging_with_subhalo={extra}{out})",
    ('SubhaloPlotter', 'subplot_detection_fits'):
        lambda r, extra, out: f"aplt.subplot_detection_fits(fit_imaging_no_subhalo=fit_no_subhalo, fit_imaging_with_subhalo={extra}{out})",
    ('SubhaloSensitivityPlotter', 'subplot_sensitivity'):
        lambda r, d, out: f"aplt.subplot_sensitivity(result={r}, data_subtracted={d}{out})",
    ('SubhaloSensitivityPlotter', 'subplot_figures_of_merit_grid'):
        lambda r, d, out: f"aplt.subplot_sensitivity_figures_of_merit(result={r}{out})",
}

# Must list longer names before shorter to avoid prefix conflicts
PLOTTER_PREFIXES = [
    'SubhaloSensitivity', 'Subhalo',
    'FitImaging', 'FitInterferometer',
    'Interferometer', 'Imaging',
    'LightProfile', 'MassProfile',
    'Galaxies', 'Galaxy',
    'Tracer',
    'Array2D', 'Grid2D',
    'Inversion', 'Mapper',
    'PointDataset',
]


class PlotterInfo:
    def __init__(self, varname, plotter_type, main_arg=None, second_arg=None,
                 extra_arg=None, output_path=None, output_format="png"):
        self.varname = varname
        self.plotter_type = plotter_type
        self.main_arg = main_arg
        self.second_arg = second_arg
        self.extra_arg = extra_arg
        self.output_path = output_path
        self.output_format = output_format
        self.pending_title = None


def parse_plotter_type(line):
    for prefix in PLOTTER_PREFIXES:
        if f'aplt.{prefix}Plotter(' in line or re.search(rf'aplt\.{prefix}Plotter\s*\(', line):
            return prefix + 'Plotter'
    if 'al.subhalo.SubhaloPlotter(' in line:
        return 'SubhaloPlotter'
    if 'al.subhalo.SubhaloSensitivityPlotter(' in line:
        return 'SubhaloSensitivityPlotter'
    return None


def extract_plotter_args(full_text, plotter_type):
    main_arg = second_arg = extra_arg = None
    if plotter_type == 'TracerPlotter':
        main_arg = extract_kwarg(full_text, 'tracer')
        second_arg = extract_kwarg(full_text, 'grid')
    elif plotter_type in ('ImagingPlotter', 'InterferometerPlotter'):
        main_arg = extract_kwarg(full_text, 'dataset')
    elif plotter_type in ('FitImagingPlotter', 'FitInterferometerPlotter'):
        main_arg = extract_kwarg(full_text, 'fit')
    elif plotter_type == 'Array2DPlotter':
        main_arg = extract_kwarg(full_text, 'array')
    elif plotter_type == 'Grid2DPlotter':
        main_arg = extract_kwarg(full_text, 'grid')
    elif plotter_type == 'LightProfilePlotter':
        main_arg = extract_kwarg(full_text, 'light_profile')
        second_arg = extract_kwarg(full_text, 'grid')
    elif plotter_type == 'MassProfilePlotter':
        main_arg = extract_kwarg(full_text, 'mass_profile')
        second_arg = extract_kwarg(full_text, 'grid')
    elif plotter_type == 'GalaxyPlotter':
        main_arg = extract_kwarg(full_text, 'galaxy')
        second_arg = extract_kwarg(full_text, 'grid')
    elif plotter_type == 'GalaxiesPlotter':
        main_arg = extract_kwarg(full_text, 'galaxies')
        second_arg = extract_kwarg(full_text, 'grid')
    elif plotter_type in ('InversionPlotter', 'MapperPlotter'):
        # Just track that we saw them; drop method calls
        pass
    elif plotter_type == 'SubhaloPlotter':
        main_arg = extract_kwarg(full_text, 'result')
        extra_arg = extract_kwarg(full_text, 'fit_imaging_with_subhalo') or 'fit_imaging_with_subhalo'
    elif plotter_type == 'SubhaloSensitivityPlotter':
        main_arg = extract_kwarg(full_text, 'result')
        second_arg = extract_kwarg(full_text, 'data_subtracted')
    return main_arg, second_arg, extra_arg


def transform_content(content):
    lines = content.split('\n')
    out = []
    mat_plot_outputs = {}
    plotters = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]

        # ── MatPlot2D assignment ─────────────────────────────────────────────
        mp_match = re.match(r'^(\s*)(\w+)\s*=\s*aplt\.MatPlot2D\s*\(', line)
        if mp_match:
            full, j = collect_balanced(lines, i)
            path_val, fmt_val = parse_output_from_mat_plot(full)
            if path_val:
                mat_plot_outputs[mp_match.group(2)] = (path_val, fmt_val)
            i = j
            continue

        # ── Plotter variable assignment ──────────────────────────────────────
        assign_match = re.match(r'^(\s*)(\w+)\s*=\s*(?:aplt\.|al\.subhalo\.)', line)
        if assign_match:
            plotter_type = parse_plotter_type(line)
            if plotter_type:
                full, j = collect_balanced(lines, i)
                var = assign_match.group(2)

                mat_var = extract_kwarg(full, 'mat_plot_2d')
                output_path, output_format = None, "png"
                if mat_var and mat_var in mat_plot_outputs:
                    output_path, output_format = mat_plot_outputs[mat_var]

                main_arg, second_arg, extra_arg = extract_plotter_args(full, plotter_type)
                plotters[var] = PlotterInfo(
                    varname=var, plotter_type=plotter_type,
                    main_arg=main_arg, second_arg=second_arg, extra_arg=extra_arg,
                    output_path=output_path, output_format=output_format,
                )
                i = j
                continue

        # ── set_title() ───────────────────────────────────────────────────────
        st_match = re.match(r'^(\s*)(\w+)\.set_title\((.+)\)\s*$', line)
        if st_match and st_match.group(2) in plotters:
            plotters[st_match.group(2)].pending_title = st_match.group(3).strip()
            i += 1
            continue

        # ── set_filename() / set_colorbar_zero_nan_values() etc. ─────────────
        sf_match = re.match(r'^(\s*)(\w+)\.(set_\w+)\s*\(', line)
        if sf_match and sf_match.group(2) in plotters:
            _, j = collect_balanced(lines, i)
            i = j
            continue

        # ── subplot_*() method calls ─────────────────────────────────────────
        sub_match = re.match(r'^(\s*)(\w+)\.(subplot_\w+)\s*\(', line)
        if sub_match:
            vi = sub_match.group(1)
            var = sub_match.group(2)
            method = sub_match.group(3)
            if var in plotters:
                full_call, j = collect_balanced(lines, i)
                p = plotters[var]
                out_str = build_output_args(p.output_path, p.output_format)
                key = (p.plotter_type, method)
                if key in SUBPLOT_MAP:
                    fn = SUBPLOT_MAP[key]
                    a1 = p.main_arg or 'result'
                    a2 = p.second_arg or p.extra_arg or '_'
                    out.append(f"{vi}{fn(a1, a2, out_str)}")
                else:
                    out.append(f"{vi}# TODO: {p.plotter_type}.{method}() -> new API")
                i = j
                continue

        # ── figures_2d() method calls ────────────────────────────────────────
        fig_match = re.match(r'^(\s*)(\w+)\.figures_2d\s*\(', line)
        if fig_match:
            vi = fig_match.group(1)
            var = fig_match.group(2)
            if var in plotters:
                full_call, j = collect_balanced(lines, i)
                p = plotters[var]
                out_str = build_output_args(p.output_path, p.output_format)
                quantities = re.findall(r'(\w+)\s*=\s*True', full_call)
                title = p.pending_title
                p.pending_title = None
                added = []
                for qty in quantities:
                    title_str = title if (title and len(quantities) == 1) else f'"{qty.replace("_", " ").title()}"'
                    qt = FIGURES_2D.get(p.plotter_type, {})
                    if qty in qt:
                        arr_fn = qt[qty]
                        arr = arr_fn(p.main_arg or 'obj', p.second_arg or 'grid')
                        if arr:
                            added.append(f"{vi}aplt.plot_array(array={arr}, title={title_str}{out_str})")
                        else:
                            # deflections etc. need multi-line — emit compute + plot
                            if qty == 'deflections_y':
                                mp = p.main_arg or 'mass'
                                g = p.second_arg or 'grid'
                                added.append(f"{vi}deflections = {mp}.deflections_yx_2d_from(grid={g})")
                                added.append(f"{vi}deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask={g}.mask)")
                                added.append(f"{vi}aplt.plot_array(array=deflections_y, title=\"Deflections Y\"{out_str})")
                            elif qty == 'deflections_x':
                                mp = p.main_arg or 'mass'
                                g = p.second_arg or 'grid'
                                added.append(f"{vi}deflections = {mp}.deflections_yx_2d_from(grid={g})")
                                added.append(f"{vi}deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask={g}.mask)")
                                added.append(f"{vi}aplt.plot_array(array=deflections_x, title=\"Deflections X\"{out_str})")
                            else:
                                added.append(f"{vi}# TODO: figures_2d({qty}=True) for {p.plotter_type}")
                    else:
                        added.append(f"{vi}# TODO: figures_2d({qty}=True) for {p.plotter_type}")
                if added:
                    out.extend(added)
                    i = j
                    continue

        # ── figure_2d() on Array2DPlotter / Grid2DPlotter ────────────────────
        fig2_match = re.match(r'^(\s*)(\w+)\.figure_2d\(\)\s*$', line)
        if fig2_match and fig2_match.group(2) in plotters:
            vi = fig2_match.group(1)
            var = fig2_match.group(2)
            p = plotters[var]
            out_str = build_output_args(p.output_path, p.output_format)
            title = p.pending_title or '""'
            p.pending_title = None
            if p.plotter_type == 'Array2DPlotter':
                out.append(f"{vi}aplt.plot_array(array={p.main_arg or 'array'}, title={title}{out_str})")
            elif p.plotter_type == 'Grid2DPlotter':
                out.append(f"{vi}aplt.plot_grid(grid={p.main_arg or 'grid'}, title={title}{out_str})")
            else:
                out.append(f"{vi}# figure_2d() for {p.plotter_type}")
            i += 1
            continue

        # ── Drop figure_figures_of_merit_grid / figure_mass_grid ─────────────
        drop_match = re.match(
            r'^(\s*)(\w+)\.(figure_figures_of_merit_grid|figure_mass_grid)\s*\(', line
        )
        if drop_match and drop_match.group(2) in plotters:
            _, j = collect_balanced(lines, i)
            i = j
            continue

        # ── Standalone Visuals2D assignment ──────────────────────────────────
        if re.match(r'^\s*\w+\s*=\s*aplt\.Visuals2D\s*\(', line):
            _, j = collect_balanced(lines, i)
            i = j
            continue

        # ── Standalone aplt.Figure/YTicks/XTicks/Title/YLabel/XLabel ─────────
        if re.match(r'^\s*\w+\s*=\s*aplt\.(Figure|YTicks|XTicks|Title|YLabel|XLabel)\s*\(', line):
            i += 1
            continue

        # ── Strip mat_plot_2d=, visuals_2d=, include_2d= from any remaining ──
        modified = line
        for kw in ('mat_plot_2d', 'visuals_2d', 'include_2d'):
            if kw + '=' in modified:
                modified = re.sub(r',\s*' + kw + r'\s*=\s*[\w\.]+(?:\([^)]*\))?', '', modified)
                modified = re.sub(kw + r'\s*=\s*[\w\.]+(?:\([^)]*\))?\s*,?\s*', '', modified)

        out.append(modified)
        i += 1

    return '\n'.join(out)


# ─── Post-processing: fix malformed calls and clean docstrings ────────────────

def post_process(content):
    # Fix malformed dataset=X.MatPlot2D() / dataset=X.Visuals2D(...) etc.
    content = re.sub(r'(dataset=\w+)\.MatPlot2D\(\)', r'\1', content)
    content = re.sub(r'(dataset=\w+)\.Visuals2D\([^)]*\)', r'\1', content)
    content = re.sub(r'(dataset=\w+)\.MatPlot2D\([^)]*\)', r'\1', content)

    # Remove # TODO: figures_2d lines that still contain old patterns
    lines = content.split('\n')
    result = []
    for line in lines:
        # Remove lines that are just TODO comments about old API
        if re.match(r'\s*# TODO: figures_2d\(.+\) for \w+Plotter', line):
            continue
        if re.match(r'\s*# TODO: \w+Plotter\.\w+\(\) -> new API', line):
            continue
        if re.match(r'\s*# TODO: \w+Plotter\.\w+ -> new API', line):
            continue
        result.append(line)
    return '\n'.join(result)


# ─── Files to skip (manual rewrites) ─────────────────────────────────────────
SKIP_FILES = {
    'scripts/guides/plot/start_here.py',
    'scripts/guides/plot/examples/visuals.py',
    'scripts/guides/plot/examples/mat_plot.py',
    'scripts/guides/plot/examples/plotters.py',
    'scripts/guides/plot/examples/searches.py',
    'scripts/guides/plot/advanced/publication/image.py',
    'scripts/guides/plot/advanced/publication/subhalo_scan.py',
    'scripts/guides/plot/advanced/multi/MultiSubplots.py',
    'scripts/guides/plot/advanced/multi/MultiYX1DPlotter.py',
    'scripts/guides/plot/advanced/multi/MultiFigurePlotter.py',
    'scripts/guides/plot/advanced/plotters_double_einstein_ring.py',
    'scripts/guides/plot/advanced/plotters_pixelization.py',
    'scripts/howtolens/chapter_1_introduction/tutorial_0_visualization.py',
}

OLD_API_MARKERS = [
    'Visuals2D', 'MatPlot2D', 'Plotter', 'figures_2d',
    'include_2d', 'visuals_2d', 'mat_plot_2d',
    'aplt.Figure', 'aplt.YTicks', 'aplt.XTicks',
    'aplt.Title', 'aplt.YLabel', 'aplt.XLabel',
]


def process_all(workspace_root):
    root = Path(workspace_root)
    changed, skipped, errors = [], [], []

    for filepath in sorted(root.rglob('*.py')):
        rel = str(filepath.relative_to(root))
        if 'transform_plotting_api' in rel:
            continue
        if rel in SKIP_FILES:
            skipped.append(rel)
            continue
        try:
            original = filepath.read_text(encoding='utf-8')
            if not any(m in original for m in OLD_API_MARKERS):
                continue
            new_content = transform_content(original)
            new_content = post_process(new_content)
            if new_content != original:
                filepath.write_text(new_content, encoding='utf-8')
                changed.append(rel)
        except Exception as e:
            errors.append((rel, str(e)))

    print(f"Changed: {len(changed)} | Skipped: {len(skipped)} | Errors: {len(errors)}")
    for f, e in errors:
        print(f"  ERROR {f}: {e}")
    return changed, skipped, errors


if __name__ == '__main__':
    workspace = sys.argv[1] if len(sys.argv) > 1 else '/home/user/autolens_workspace'
    process_all(workspace)
