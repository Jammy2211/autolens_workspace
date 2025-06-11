def fit():
    """
    slam_graph (Source, Light and Mass): Source Light Pixelized + Light Profile + Mass Total + Subhalo NFW
    ================================================================================================

    slam_graph pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
    lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
    which customize the model and analysis in that pipeline.

    The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
    uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

    Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS PIPELINE and SUBHALO PIPELINE this slam_graph script
    fits `Imaging` of a strong lens system, where in the final model:

     - The lens galaxy's light is a bulge+disk `Sersic` and `Exponential`.
     - The lens galaxy's total mass distribution is an `Isothermal`.
     - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
     - The source galaxy is an `Inversion`.

    This uses the slam_graph pipelines:

     `source_lp`
     `source_pix`
     `light_lp`
     `mass_total`
     `subhalo/detection`

    Check them out for a full description of the analysis!
    """
    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

    import numpy as np
    import os
    from os import path

    import autofit as af
    import autolens as al
    import autolens.plot as aplt
    import slam_multi

    """
    __Dataset__ 

    Load, plot and mask the `Imaging` data.
    """
    dataset_waveband_list = ["g", "r"]
    pixel_scale_list = [0.12, 0.08]

    dataset_name = "lens_sersic"
    dataset_main_path = path.join("dataset", "multi", dataset_name)
    dataset_path = path.join(dataset_main_path, dataset_name)

    dataset_list = []

    for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):
        dataset = al.Imaging.from_fits(
            data_path=path.join(dataset_main_path, f"{dataset_waveband}_data.fits"),
            noise_map_path=path.join(
                dataset_main_path, f"{dataset_waveband}_noise_map.fits"
            ),
            psf_path=path.join(dataset_main_path, f"{dataset_waveband}_psf.fits"),
            pixel_scales=pixel_scale,
        )

        mask_radius = 3.0

        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
        )

        dataset = dataset.apply_mask(mask=mask)

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[8, 4, 1],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )

        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        dataset_list.append(dataset)

    """
    __Settings AutoFit__

    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    settings_search = af.SettingsSearch(
        path_prefix=path.join("slam_graph", "base"),
        number_of_cores=1,
        session=None,
    )

    """
    __Redshifts__

    The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
    from arc-seconds to kiloparsecs, masses to solar masses, etc.).
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __SOURCE LP PIPELINE__

    The SOURCE LP PIPELINE fits an identical to the `start_here.ipynb` example, except:

     - The model includes the (y,x) offset of each dataset relative to the first dataset, which is added to every
      `AnalysisImaging` object such that there are 2 extra parameters fitted for each dataset.
    """

    # Lens Light

    centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    total_gaussians = 30
    gaussian_per_basis = 2

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    lens_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    # Source Light

    centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    total_gaussians = 30
    gaussian_per_basis = 1

    log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    source_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

    source_lp_result = slam_multi.source_lp.run(
        settings_search=settings_search,
        analysis_list=analysis_list,
        lens_bulge=lens_bulge,
        lens_disk=None,
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=source_bulge,
        mass_centre=(0.0, 0.0),
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
        dataset_model=af.Model(al.DatasetModel),
    )

    """
    __SOURCE PIX PIPELINE__

    The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
    reconstructs the source galaxy's light. 

    This pixelization adapts its source pixels to the morphology of the source, placing more pixels in its 
    brightest regions. To do this, an "adapt image" is required, which is the lens light subtracted image meaning
    only the lensed source emission is present.

    The SOURCE LP Pipeline result is not good enough quality to set up this adapt image (e.g. the source
    may be more complex than a simple light profile). The first step of the SOURCE PIX PIPELINE therefore fits a new
    model using a pixelization to create this adapt image.

    The first search, which is an initialization search, fits an `Overlay` image-mesh, `Delaunay` mesh 
    and `AdaptiveBrightnessSplit` regularization.

    __Adapt Images / Image Mesh Settings__

    If you are unclear what the `adapt_images` and `SettingsInversion` inputs are doing below, refer to the 
    `autolens_workspace/*/imaging/advanced/chaining/pix_adapt/start_here.py` example script.

    __Settings__:

     - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
     in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
    """
    positions_likelihood = source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    )

    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_image_maker=al.AdaptImageMaker(result=result),
            positions_likelihood_list=[positions_likelihood],
        )
        for result in source_lp_result
    ]

    source_pix_result_1 = slam_multi.source_pix.run_1(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_lp_result=source_lp_result,
        mesh_init=al.mesh.Delaunay,
        dataset_model=af.Model(al.DatasetModel),
    )

    """
    __SOURCE PIX PIPELINE 2 (with lens light)__

    The second search, which uses the mesh and regularization used throughout the remainder of the slam_graph pipelines,
    fits the following model:

    - Uses a `Hilbert` image-mesh. 

    - Uses a `Delaunay` mesh.

     - Uses an `AdaptiveBrightnessSplit` regularization.

     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
     SOURCE PIX PIPELINE.

    The `Hilbert` image-mesh and `AdaptiveBrightness` regularization adapt the source pixels and regularization weights
    to the source's morphology.

    Below, we therefore set up the adapt image using this result.
    """
    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_image_maker=al.AdaptImageMaker(result=result),
            settings_inversion=al.SettingsInversion(
                image_mesh_min_mesh_pixels_per_pixel=3,
                image_mesh_min_mesh_number=5,
                image_mesh_adapt_background_percent_threshold=0.1,
                image_mesh_adapt_background_percent_check=0.8,
            ),
        )
        for result in source_pix_result_1
    ]

    source_pix_result_2 = slam_multi.source_pix.run_2(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        image_mesh=al.image_mesh.Hilbert,
        mesh=al.mesh.Delaunay,
        regularization=al.reg.AdaptiveBrightnessSplit,
        dataset_model=af.Model(al.DatasetModel),
    )

    """
    __LIGHT LP PIPELINE__

    The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
    lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
    In this example it:

     - Uses a multi Gaussian expansion with 2 sets of 30 Gaussians for the lens galaxy's light. [6 Free Parameters].

     - Uses an `Isothermal` mass model with `ExternalShear` for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].

     - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

     - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS PIPELINE [fixed values].   
    """
    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_image_maker=al.AdaptImageMaker(result=result),
            raise_inversion_positions_likelihood_exception=False,
        )
        for result in source_pix_result_1
    ]

    centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
    centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)

    total_gaussians = 30
    gaussian_per_basis = 2

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    lens_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    light_result = slam_multi.light_lp.run(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=lens_bulge,
        lens_disk=None,
        dataset_model=af.Model(al.DatasetModel),
    )

    """
    __MASS TOTAL PIPELINE__

    The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
    using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors and the lens 
    light model of the LIGHT LP PIPELINE. 

    In this example it:

     - Uses a linear Multi Gaussian Expansion bulge [fixed from LIGHT LP PIPELINE].

     - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
     PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].

     - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

     - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.

    __Settings__:

     - adapt: We may be using adapt features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
     hyper dataset if required.

     - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
     in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
    """
    positions_likelihood = source_pix_result_1[0].positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    )

    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_image_maker=al.AdaptImageMaker(result=result),
            positions_likelihood_list=[positions_likelihood],
        )
        for result in source_pix_result_1
    ]

    mass_result = slam_multi.mass_total.run(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
        dataset_model=af.Model(al.DatasetModel),
    )

    """
    __SUBHALO PIPELINE (single plane detection)__

    The SUBHALO PIPELINE (single plane detection) consists of the following searches:

     1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a 
     subhalo. This uses the same model as fitted in the MASS PIPELINE. 
     2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo. 
     3) If there is a successful detection a final search is performed to refine its parameters.

    For this runner the SUBHALO PIPELINE customizes:

     - The [number_of_steps x number_of_steps] size of the grid-search, as well as the dimensions it spans in arc-seconds.
     - The `number_of_cores` used for the gridsearch, where `number_of_cores > 1` performs the model-fits in paralle using
     the Python multiprocessing module.
    """
    analysis_list = [
        al.AnalysisImaging(
            dataset=result.max_log_likelihood_fit.dataset,
            adapt_image_maker=al.AdaptImageMaker(result=result),
            positions_likelihood_list=[positions_likelihood],
        )
        for result in source_pix_result_1
    ]

    subhalo_result_1 = slam_multi.subhalo.detection.run_1_no_subhalo(
        settings_search=settings_search,
        analysis_list=analysis_list,
        mass_result=mass_result,
    )

    subhalo_grid_search_result_2 = slam_multi.subhalo.detection.run_2_grid_search(
        settings_search=settings_search,
        analysis_list=analysis_list,
        mass_result=mass_result,
        subhalo_result_1=subhalo_result_1,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
        grid_dimension_arcsec=3.0,
        number_of_steps=2,
    )

    subhalo_result_3 = slam_multi.subhalo.detection.run_3_subhalo(
        settings_search=settings_search,
        analysis_list=analysis_list,
        subhalo_result_1=subhalo_result_1,
        subhalo_grid_search_result_2=subhalo_grid_search_result_2,
        subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    )

    """
    Finish.
    """


if __name__ == "__main__":

    fit()
