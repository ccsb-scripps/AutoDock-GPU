#!/usr/bin/env python

import argparse
import sys
import os
import pybel
import autodockdev
from autodockdev import motions
import nlopt

import numpy as np # debug only

class MyParser(argparse.ArgumentParser):
    """display full help message if parser finds error"""
    def error(self, message):
        self.print_help()
        sys.stderr.write('\nERROR:\n  %s\n' % message)
        sys.exit(2)

def get_args():
    parser = MyParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('lig', help='ligand')
    parser.add_argument('--fld', help='maps .fld', required=True)
    args = parser.parse_args()
    return args

class DockingSystem():
    """ set up a system with forcefield, maps, etc... """

    def __init__(self, coordsobj, zmol, mapo=None):
        """ """

        # pairwise scorer a.k.a interaction handler
        energy_scorer = autodockdev.PairwiseEnergy()
        force_scorer = autodockdev.PairwiseDerivatives()
        self.ff = autodockdev.AutoDockParameters()
        self.H = autodockdev.InteractionHandler(
            [zmol], energy_scorer, force_scorer, self.ff)

        # maps
        self.mapo = mapo

        # ligand coordinates
        self.coordsobj = coordsobj

        self.n_evals = 0 
        self.minsofar = float('+inf')

    def eval(self, genes, grad=None):
        """ 
            Input:  genes
            Output: score
        """ 

        # set genes
        autodockdev.set_genes(self.coordsobj, genes)
        self.coordsobj.zmol.write_xyz('running.xyz', self.coordsobj, 'a')

        # communicate new coordinates to interaction handler
        self.H.download_coords_from(self.coordsobj)
       
        # calc pairwise distances
        self.H.calc_vectors()
        self.H.calc_sqr_dists()
        self.H._calc_distances()
         
        energy = 0.

        # get pairwise energy
        self.H.calc_energy_terms()
        energy += sum(self.H.energy['ad4_vdw']) * self.ff.weight_vdw
        energy += sum(self.H.energy['ad4_hb'])  * self.ff.weight_hb
        energy += sum(self.H.energy['elec'])    * self.ff.weight_elec
        energy += sum(self.H.energy['ad4_dsol'])* self.ff.weight_dsol

        # add maps energy
        energy_maps, grad_maps = mapo.calc_energy_and_forces(self.coordsobj)
        energy += energy_maps

        # pairwise forces | ommiting: H.calc_dihedral_forces()
        if type(grad) != type(None):

            self.H.calc_forces() 
            self.H.upload_forces_to(self.coordsobj)

            # map forces TODO use mapo.calc_forces2 TODO
            self.coordsobj.forces -= grad_maps#mapo.calc_forces(self.coordsobj)

            # delta genes (quaternion, not cube3)
            delta_g, qsf = autodockdev.forces_to_delta_genes(self.coordsobj)

            # convert "quaternion force" to gradient in cube3
            grad_u = motions._get_cube3_gradient(self.coordsobj, delta_g[3:7]) * qsf
            delta_g = np.hstack((delta_g[0:3], grad_u, delta_g[7:]))

            # update grad "in-place" (this is an NLOPT requirement)
            for i in range(len(grad)):
                grad[i] = -delta_g[i]

        self.n_evals += 1
        #print 'iter #', self.n_evals
        #print 'gradient:', grad
        #print 'genes:', genes
        self.minsofar = min(self.minsofar, energy)
        #print '%6d, %12.1f, %8.3f' % (self.n_evals, energy, self.minsofar)#, genes

        
        return float(energy) # float necessary to avoid ValueError: nlopt invalid argument

TWOPI = np.pi * 2

args = get_args()

# lig
lig, ext = os.path.splitext(args.lig)
ext = ext.replace('.', '') 
mol = pybel.readfile(ext, '%s.%s' % (lig, ext)).next()
zmol = autodockdev.obmol2zmol(mol)

dc = autodockdev.DynamicCoords(zmol, zmol.coords, zmol.about)

# maps
mapo = autodockdev.AutoDockMaps(args.fld)

# docking system
ds = DockingSystem(dc, zmol, mapo)


#g0 = [0.5*lower_lim[i] + 0.5 * upper_lim[i] for i in range(3)] +  [1E-12, 0.25*TWOPI, 0.] + [0.]*ntor
#g0 = mapo.center +  [1E-12, 0.25*TWOPI, 0.] + [0.]*ntor
ntor = len(zmol.tors)
genotype = list(dc.about) +  [0.1, 0.75*TWOPI, 0.5*TWOPI] + [0.]*ntor
ana_gradient = [0. for g in genotype] 

energy = ds.eval(genotype, ana_gradient)


print 'energy:', energy
print 'analytic derivatives:                 %10.4f %10.4f %10.4f %20.4f %20.4f %20.4f' % tuple(ana_gradient)

dx_list = [1E-6, 1E-8, 1E-10]
for dx in dx_list:
    num_gradient = [0. for g in genotype]
    for i in range(6):
        new_genotype = genotype[:]
        new_genotype[i] += dx
        new_energy = ds.eval(new_genotype)
        num_gradient[i] = (new_energy - energy) / dx
    
    print 'numerical derivatives (%.10f): %10.4f %10.4f %10.4f %20.4f %20.4f %20.4f' % tuple([dx] + num_gradient)




