println()
@info "SolidStateDetectors Benchmark"
println()
versioninfo()
println()
using SolidStateDetectors
using BenchmarkTools

T = Float32
configfilename = SolidStateDetectors.SSD_examples[:InvertedCoax];
sim = Simulation{T}(SSD_examples[:Coax]);
detector = sim.detector
init_grid_size = (10, 10, 10)
init_grid_spacing = missing
grid = SolidStateDetectors.Grid(detector, init_grid_size = init_grid_size, init_grid_spacing = init_grid_spacing)
convergence_limit = T(5e-6)
max_refinements = 3
refinement_limits = T[1e-4, 1e-4, 1e-4]
min_grid_spacing = T[1e-4, 1e-4, 1e-4]
depletion_handling = false
use_nthreads = parse(Int, ENV["JULIA_NUM_THREADS"])
sor_consts = T[1.4, 1.85]
max_n_iterations = 10000
verbose = false
n_iterations_between_checks = 500
refine = max_refinements > 0 ? true : false
only_2d = length(grid.axes[2]) == 1 ? true : false
# calculate_electric_potential!(sim, init_grid_size = (40, 10, 40),
#                                 use_nthreads = 1, max_refinements = 0)

fssrb = SolidStateDetectors.PotentialSimulationSetupRB(detector, grid);

gw1 = fssrb.geom_weights[1].weights  # r or x
gw2 = fssrb.geom_weights[2].weights  # φ or y
gw3 = fssrb.geom_weights[3].weights  # z or z

even_points = true
update_even_points = Val{even_points}()
depletion_handling = Val{false}()
rb_tar_idx, rb_src_idx = even_points ? (SSD.rb_even, SSD.rb_odd) : (SSD.rb_odd, SSD.rb_even)
# for idx3 in 2:(size(fssrb.potential, 3) - 1)
idx3 = 2
bulk_is_ptype = Val{fssrb.bulk_is_ptype}()
is_weighting_potential = Val{false}()

SolidStateDetectors.innerloops!(idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3,
            fssrb, update_even_points, depletion_handling,
            bulk_is_ptype, is_weighting_potential)
@info "Innerloop:"
@btime SolidStateDetectors.innerloops!(idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3,
            fssrb, update_even_points, depletion_handling,
            bulk_is_ptype, is_weighting_potential)




SolidStateDetectors.update!(fssrb, use_nthreads, update_even_points,
                            depletion_handling, bulk_is_ptype, is_weighting_potential)
@info "update!: $(use_nthreads) threads"
@btime SolidStateDetectors.update!(fssrb, use_nthreads, update_even_points,
                            depletion_handling, bulk_is_ptype, is_weighting_potential)

# @code_llvm SolidStateDetectors.innerloops!(idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3,
#             fssrb, update_even_points, depletion_handling,
#             bulk_is_ptype, is_weighting_potential)
# @code_warntype SolidStateDetectors.innerloops!(idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3,
#             fssrb, update_even_points, depletion_handling,
#             bulk_is_ptype, is_weighting_potential)
# @btime SolidStateDetectors.innerloops!(idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3,
#             fssrb, update_even_points, depletion_handling,
#             bulk_is_ptype, is_weighting_potential)
