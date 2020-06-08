/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/
#ifndef FILELIST_HPP
#define FILELIST_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class FileList{

	public:

	bool used;
	bool preload_maps;
	bool maps_are_loaded;
	bool load_maps_gpu;
	char filename [128];
	int nfiles;
	std::vector<std::string> resnames;
	std::vector<std::string> fld_files;
	std::vector<std::string> ligand_files;

	// Default to unused, with 1 file
	FileList() : used( false ), nfiles( 1 ), preload_maps( false ), maps_are_loaded( false ), load_maps_gpu( false ) {}
};

#endif
