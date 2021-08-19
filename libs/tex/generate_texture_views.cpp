/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <util/strings.h>
#include <util/timer.h>
#include <util/tokenizer.h>
#include <core/image_io.h>
#include <core/image_tools.h>
#include <core/bundle_io.h>
#include <core/scene.h>

#include "progress_counter.h"
#include "texturing.h"

TEX_NAMESPACE_BEGIN

void
from_mve_scene(std::string const & scene_dir, std::string const & embedding,
    std::vector<TextureView> * texture_views, unsigned int resolution) {

    mve::Scene::Ptr scene;
    try {
        scene = mve::Scene::create(scene_dir);
    } catch (std::exception& e) {
        std::cerr << "Could not open scene: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::size_t num_views = scene->get_views().size();
    texture_views->reserve(num_views);

    bool using_level = embedding.find("-L") != std::string::npos;
    if (using_level && resolution != 0) {
      std::cout << "Warning: ignoring resolution parameter because an embedding with a level was given" << std::endl;
    }

    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < num_views; ++i) {
        std::string used_embedding = embedding;

        mve::View::Ptr view = scene->get_view_by_id(i);
        if (view == NULL) {
            continue;
        }

        if (!view->has_image(used_embedding, mve::IMAGE_TYPE_UINT8)) {
            if (embedding == "original" && view->has_image("undistorted")) {
                std::cout << "Warning: View " << view->get_name() << " did not have embedding \"original\""
                    << " but does have \"undistorted\" which will be used instead" << std::endl;
                used_embedding = "undistorted";
            } else if (embedding == "undistorted" && view->has_image("original")) {
                std::cout << "Undistorting view " << view->get_name() << std::endl;
                used_embedding = "original";
            } else {
                std::cout << "Warning: View " << view->get_name() << " has no byte image "
                    << used_embedding << std::endl;
                continue;
            }
        }

        if (!using_level) {
            const std::string target_embedding = "undist-L" + std::to_string(resolution);
            if (!view->has_image(target_embedding, mve::IMAGE_TYPE_UINT8)) {
                #pragma omp critical
                { std::cout << "Rescaling view " << view->get_name() << std::endl; }
                const mve::CameraInfo& camera = view->get_camera();
                mve::ByteImage::Ptr image = view->get_byte_image(used_embedding);
                if (used_embedding == "original") {
                    image = mve::image::image_undistort_k2k4<uint8_t>(image,
                        camera.flen, camera.dist[0], camera.dist[1]);
                }
                for (unsigned int r = 0; r < resolution; ++r) {
                    image = mve::image::rescale_half_size_gaussian<uint8_t>(image);
                }
                view->set_image(image, target_embedding);
                view->save_view();
            }
            used_embedding = "undist-L" + std::to_string(resolution);
        }

        mve::View::ImageProxy const * image_proxy = view->get_image_proxy(used_embedding);
        if (image_proxy->channels < 3) {
            std::cout << "Warning: Image " << used_embedding << " of view " <<
                view->get_name() << " is not a color image!" << std::endl;
            continue;
        }

        #pragma omp critical
        {
            texture_views->push_back(TextureView(view->get_id(), view->get_camera(), util::fs::abspath(
                    util::fs::join_path(view->get_directory(), image_proxy->filename))));
        }
    }
}

void
from_images_and_camera_files(std::string const & path, std::vector<TextureView> * texture_views,
    std::string const & tmp_dir, unsigned int resolution)
{
    util::fs::Directory dir(path);
    std::sort(dir.begin(), dir.end());
    std::vector<std::string> files;
    for (std::size_t i = 0; i < dir.size(); ++i) {
        util::fs::File const & cam_file = dir[i];
        if (cam_file.is_dir) continue;

        std::string cam_file_ext = util::string::uppercase(util::string::right(cam_file.name, 4));
        if (cam_file_ext != ".CAM") continue;

        std::string prefix = util::string::left(cam_file.name, cam_file.name.size() - 4);
        if (prefix.empty()) continue;

        /* Find corresponding image file. */
        int step = 1;
        for (std::size_t j = i + 1; ; j += step) {

            /* Since the files are sorted we can break - no more files with the same prefix exist. */
            if (j >= dir.size() || util::string::left(dir[j].name, prefix.size()) != prefix) {
                if (step == 1) {
                    j = i;
                    step = -1;
                    continue;
                } else {
                    break;
                }
            }
            util::fs::File const & img_file = dir[j];

            /* Image file (based on extension)? */
            std::string img_file_ext = util::string::uppercase(util::string::right(img_file.name, 4));
            if (img_file_ext != ".PNG" && img_file_ext != ".JPG" &&
                img_file_ext != "TIFF" && img_file_ext != "JPEG") continue;

            files.push_back(cam_file.get_absolute_name());
            files.push_back(img_file.get_absolute_name());
            break;
        }
    }

    ProgressCounter view_counter("\tLoading", files.size() / 2);
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < files.size(); i += 2) {
        view_counter.progress<SIMPLE>();
        const std::string cam_file = files[i];
        const std::string img_file = files[i + 1];

        /* Read CAM file. */
        std::ifstream infile(cam_file.c_str(), std::ios::binary);
        if (!infile.good()) {
            throw util::FileException(util::fs::basename(cam_file), std::strerror(errno));
        }
        std::string cam_int_str, cam_ext_str;
        std::getline(infile, cam_ext_str);
        std::getline(infile, cam_int_str);
        util::Tokenizer tok_ext, tok_int;
        tok_ext.split(cam_ext_str);
        tok_int.split(cam_int_str);
        #pragma omp critical
        if (tok_ext.size() != 12 || tok_int.size() < 1) {
            std::cerr << "Invalid CAM file: " << util::fs::basename(cam_file) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        /* Create cam_info and eventually undistort image. */
        mve::CameraInfo cam_info;
        cam_info.set_translation_from_string(tok_ext.concat(0, 3));
        cam_info.set_rotation_from_string(tok_ext.concat(3, 0));

        std::stringstream ss(cam_int_str);
        ss >> cam_info.flen;
        if (ss.peek() && !ss.eof())
            ss >> cam_info.dist[0];
        if (ss.peek() && !ss.eof())
            ss >> cam_info.dist[1];
        if (ss.peek() && !ss.eof())
            ss >> cam_info.paspect;
        if (ss.peek() && !ss.eof())
            ss >> cam_info.ppoint[0];
        if (ss.peek() && !ss.eof())
            ss >> cam_info.ppoint[1];

        std::string image_file = util::fs::abspath(img_file);
        if (cam_info.dist[0] != 0.0f) {
            mve::ByteImage::Ptr image = mve::image::load_file(img_file);
            if (cam_info.dist[1] != 0.0f) {
                image = mve::image::image_undistort_k2k4<uint8_t>(image,
                    cam_info.flen, cam_info.dist[0], cam_info.dist[1]);
            } else {
                image = mve::image::image_undistort_vsfm<uint8_t>(image,
                    cam_info.flen, cam_info.dist[0]);
            }

            for (unsigned int r = 0; r < resolution; ++r) {
                mve::image::rescale_half_size_gaussian<uint8_t>(image);
            }

            image_file = util::fs::join_path(
                tmp_dir,
                util::fs::replace_extension(util::fs::basename(img_file), "png")
            );
            mve::image::save_png_file(image, image_file);
        }

        #pragma omp critical
        texture_views->push_back(TextureView(i / 2, cam_info, image_file));

        view_counter.inc();
    }
}

void
from_nvm_scene(std::string const & nvm_file, std::vector<TextureView> * texture_views,
    std::string const & tmp_dir, unsigned int resolution)
{
    std::vector<mve::AdditionalCameraInfo> nvm_cams;
    mve::Bundle::Ptr bundle = mve::load_nvm_bundle(nvm_file, &nvm_cams);
    mve::Bundle::Cameras& cameras = bundle->get_cameras();

    ProgressCounter view_counter("\tLoading", cameras.size());
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        view_counter.progress<SIMPLE>();
        mve::CameraInfo& mve_cam = cameras[i];
        mve::AdditionalCameraInfo const& nvm_cam = nvm_cams[i];

        mve::ByteImage::Ptr image = mve::image::load_file(nvm_cam.filename);

        int const maxdim = std::max(image->width(), image->height());
        mve_cam.flen = mve_cam.flen / static_cast<float>(maxdim);

        image = mve::image::image_undistort_vsfm<uint8_t>
            (image, mve_cam.flen, nvm_cam.radial_distortion);

        for (unsigned int r = 0; r < resolution; ++r) {
            mve::image::rescale_half_size_gaussian<uint8_t>(image);
        }

        const std::string image_file = util::fs::join_path(
            tmp_dir,
            util::fs::replace_extension(
                util::fs::basename(nvm_cam.filename),
                "png"
            )
        );
        mve::image::save_png_file(image, image_file);

        #pragma omp critical
        texture_views->push_back(TextureView(i, mve_cam, image_file));

        view_counter.inc();
    }
}

void
generate_texture_views(std::string const & in_scene, std::vector<TextureView> * texture_views,
    std::string const & tmp_dir, unsigned int resolution)
{
    /* Determine input format. */

    /* BUNDLEFILE */
    if (util::fs::file_exists(in_scene.c_str())) {
        std::string const & file = in_scene;
        std::string extension = util::string::uppercase(util::string::right(file, 3));
        if (extension == "NVM") {
            from_nvm_scene(file, texture_views, tmp_dir, resolution);
        }
    }

    /* SCENE_FOLDER */
    if (util::fs::dir_exists(in_scene.c_str())) {
        from_images_and_camera_files(in_scene, texture_views, tmp_dir, resolution);
    }

    /* MVE_SCENE::EMBEDDING */
    size_t pos = in_scene.rfind("::");
    if (pos != std::string::npos) {
        std::string scene_dir = in_scene.substr(0, pos);
        std::string image_name = in_scene.substr(pos + 2, in_scene.size());
        from_mve_scene(scene_dir, image_name, texture_views, resolution);
    }

    std::sort(texture_views->begin(), texture_views->end(),
        [] (TextureView const & l, TextureView const & r) -> bool {
            return l.get_id() < r.get_id();
        }
    );

    std::size_t num_views = texture_views->size();
    if (num_views == 0) {
        std::cerr
            << "No proper input scene descriptor given.\n"
            << "A input descriptor can be:\n"
            << "BUNDLE_FILE - a bundle file (currently onle .nvm files are supported)\n"
            << "SCENE_FOLDER - a folder containing images and .cam files\n"
            << "MVE_SCENE::EMBEDDING - a mve scene and embedding\n"
            << std::endl;
        exit(EXIT_FAILURE);
    }
}

TEX_NAMESPACE_END
