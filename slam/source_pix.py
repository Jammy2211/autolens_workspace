import autofit as af
import autolens as al

from typing import Tuple, Union


def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
    source_lp_results: af.ResultsCollection,
    image_mesh_init: af.Model(al.AbstractImageMesh) = af.Model(al.image_mesh.Overlay),
    mesh_init: af.Model(al.AbstractMesh) = af.Model(al.mesh.Delaunay),
    image_mesh_init_shape: Tuple[int, int] = (34, 34),
    regularization_init: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
    image_mesh: af.Model(al.AbstractImageMesh) = af.Model(al.image_mesh.Hilbert),
    mesh: af.Model(al.AbstractMesh) = af.Model(al.mesh.Delaunay),
    regularization: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
) -> af.ResultsCollection:
    """
    The SLaM SOURCE PIX PIPELINE, which initializes a lens model which uses a pixelized source for the source
    analysis.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_adapt
        The setup of the adapt fit.
    source_lp_results
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    image_mesh_init
        The image mesh, which defines how the mesh centres are computed in the image-plane, used by the pixelization
        in the first search which initializes the source.
    image_mesh_init_shape
        The shape (e.g. resolution) of the image-mesh used in the initialization search (`search[1]`). This is only
        used if the image-mesh has a `shape` parameter (e.g. `Overlay`).
    mesh_init
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the first search which initializes the source.
    regularization_init
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        which fits the source light in the initialization search (`search[1]`).
    image_mesh
        The image mesh, which defines how the mesh centres are computed in the image-plane, used by the pixelization
        in the final search which improves the source adaption.
    mesh
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the final search which improves the source adaption.
    regularization
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        in the final search which improves the source adaption.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using a light profiles [parameters fixed to result of SOURCE LP PIPELINE].

     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE LP PIPELINE].

     - The source galaxy's light is the input initialization imagemesh, mesh and regularization scheme [parameters of 
     regularization free to vary].

    This search improves the lens mass model by modeling the source using a `Pixelization` and computes the adapt
    images that are used in search 2.
    """

    analysis.adapt_images = source_lp_results.last.adapt_images

    mass = al.util.chaining.mass_from(
        mass=source_lp_results.last.model.galaxies.lens.mass,
        mass_result=source_lp_results.last.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    image_mesh_init.shape = image_mesh_init_shape

    model_1 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_results.last.instance.galaxies.lens.redshift,
                bulge=source_lp_results.last.instance.galaxies.lens.bulge,
                disk=source_lp_results.last.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_lp_results.last.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_results.last.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    image_mesh=image_mesh_init,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
        clumps=al.util.chaining.clumps_from(result=source_lp_results.last),
    )

    search_1 = af.Nautilus(
        name="source_pix[1]_light[fixed]_mass[init]_source[pix_init_mag]",
        **settings_search.search_dict,
        n_live=150,
    )

    result_1 = search_1.fit(
        model=model_1, analysis=analysis, **settings_search.fit_dict
    )

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    Search 2 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using a light profiles [parameters fixed to result of SOURCE LP PIPELINE].
    - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
    - The source galaxy's light is the input final mesh and regularization.

    This search initializes the pixelization's mesh and regularization.
    """

    analysis.adapt_images = result_1.adapt_images

    model_2 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_results.last.instance.galaxies.lens.redshift,
                bulge=source_lp_results.last.instance.galaxies.lens.bulge,
                disk=source_lp_results.last.instance.galaxies.lens.disk,
                mass=result_1.instance.galaxies.lens.mass,
                shear=result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_results.last.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    image_mesh=image_mesh,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
        ),
        clumps=al.util.chaining.clumps_from(result=source_lp_results.last),
    )

    if setup_adapt.mesh_pixels_fixed is not None:
        if hasattr(model_2.galaxies.source.pixelization.image_mesh, "pixels"):
            model_2.galaxies.source.pixelization.image_mesh.pixels = (
                setup_adapt.mesh_pixels_fixed
            )

    search_2 = af.DynestyStatic(
        name="source_pix[2]_light[fixed]_mass[fixed]_source[pix]",
        **settings_search.search_dict,
        nlive=100,
    )

    result_2 = search_2.fit(
        model=model_2, analysis=analysis, **settings_search.fit_dict
    )

    return af.ResultsCollection([result_1, result_2])
