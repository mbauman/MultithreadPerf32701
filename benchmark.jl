println()
@info "SolidStateDetectors Benchmark"
println()
versioninfo()
println()
using Serialization
using BenchmarkTools

const rb_even = Int(2)
const rb_odd  = Int(1)

const rb_bool_even = true
const rb_bool_odd  = false
const update_bit   = 0x01
@inline function nidx( rbidx::Int, ::Val{true}, ::Val{true})::Int
   return (rbidx - 1) * 2
end
@inline function nidx( rbidx::Int, ::Val{true}, ::Val{false})::Int
   return (rbidx - 1) * 2 - 1
end
@inline function nidx( rbidx::Int, ::Val{true}, b::Bool)::Int
   return (rbidx - 1) * 2 - !b
end
@inline function nidx( rbidx::Int, ::Val{false}, ::Val{true})::Int
   return (rbidx - 1) * 2 - 1
end
@inline function nidx( rbidx::Int, ::Val{false}, ::Val{false})::Int
   return (rbidx - 1) * 2
end
@inline function nidx( rbidx::Int, ::Val{false}, b::Bool)::Int
   return (rbidx - 1) * 2 - b
end

"""
    update!(fssrb::PotentialSimulationSetupRB{T, 3, 4, S}, RBT::DataType)::Nothing

Loop over `even` grid points. A point is `even` if the sum of its cartesian indicies (of the not extended grid) is even.
Even points get the red black index (rbi) = 2. ( -> rbpotential[ inds..., rbi ]).
"""
@fastmath function update!( fssrb, use_nthreads::Int,
                            update_even_points::Val{even_points},
                            depletion_handling::Val{depletion_handling_enabled},
                            bulk_is_ptype::Val{_bulk_is_ptype},
                            is_weighting_potential::Val{_is_weighting_potential})::Nothing where {even_points, depletion_handling_enabled, _bulk_is_ptype, _is_weighting_potential}
    @inbounds begin
        rb_tar_idx::Int, rb_src_idx::Int = even_points ? (rb_even::Int, rb_odd::Int) : (rb_odd::Int,rb_even::Int)

        gw1 = fssrb.geom_weights[1].weights  # r or x
        gw2 = fssrb.geom_weights[2].weights  # φ or y
        gw3 = fssrb.geom_weights[3].weights  # z or z

        # for idx3 in 2:(size(fssrb.potential, 3) - 1)
        Base.Threads.@threads for idx3 in 2:(size(fssrb.potential, 3) - 1)
        # @onthreads 1:use_nthreads for idx3 in workpart(2:(size(fssrb.potential, 3) - 1), 1:use_nthreads, Base.Threads.threadid())
            innerloops!( idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3, fssrb, update_even_points, depletion_handling, bulk_is_ptype, is_weighting_potential)
        end
    end
    nothing
end

"""
    innerloops!(  ir::Int, rb_tar_idx::Int, rb_src_idx::Int, gw_r::Array{T, 2}, gw_φ::Array{T, 2}, gw_z::Array{T, 2}, fssrb::PotentialSimulationSetupRB{T, 3, 4, :cylindrical},
                                update_even_points::Val{even_points},
                                depletion_handling::Val{depletion_handling_enabled},
                                bulk_is_ptype::Val{_bulk_is_ptype}  )::Nothing where {T, even_points, depletion_handling_enabled, _bulk_is_ptype}

(Vectorized) inner loop for Cylindrical coordinates. This function does all the work in the field calculation.
"""
@fastmath function innerloops!( ir::Int, rb_tar_idx::Int, rb_src_idx::Int, gw_r::Array{T, 2}, gw_φ::Array{T, 2}, gw_z::Array{T, 2}, fssrb,
                                update_even_points::Val{even_points},
                                depletion_handling::Val{depletion_handling_enabled},
                                bulk_is_ptype::Val{_bulk_is_ptype},
                                is_weighting_potential::Val{_is_weighting_potential}  )::Nothing where {T, even_points, depletion_handling_enabled, _bulk_is_ptype, _is_weighting_potential}
    @inbounds begin
        inr::Int = ir - 1

        pwwrr::T               = gw_r[1, inr]
        pwwrl::T               = gw_r[2, inr]
        r_inv_pwΔmpr::T        = gw_r[3, inr]
        Δr_ext_inv_r_pwmprr::T = gw_r[4, inr]
        Δr_ext_inv_l_pwmprl::T = gw_r[5, inr]
        Δmpr_squared::T        = gw_r[6, inr]

        for iφ in 2:(size(fssrb.potential, 2) - 1)
            # φ loop
            inφ::Int = iφ - 1
            pwwφr::T        = gw_φ[1, inφ]
            pwwφl::T        = gw_φ[2, inφ]
            pwΔmpφ::T       = gw_φ[3, inφ]
            Δφ_ext_inv_r::T = gw_φ[4,  iφ]
            Δφ_ext_inv_l::T = gw_φ[4, inφ]

            if inr == 1
                pwwφr = T(0.5)
                pwwφl = T(0.5)
                pwΔmpφ = T(2π)
                Δφ_ext_inv_r = inv(pwΔmpφ)
                Δφ_ext_inv_l = Δφ_ext_inv_r
            end

            rφi_is_even::Bool = iseven(ir + iφ)
            # rφi_is_even_t::Val{rφi_is_even} = Val{rφi_is_even}()

            pwwrr_pwwφr::T = pwwrr * pwwφr
            pwwrr_pwwφl::T = pwwrr * pwwφl
            pwwrl_pwwφr::T = pwwrl * pwwφr
            pwwrl_pwwφl::T = pwwrl * pwwφl

            Δr_ext_inv_r_pwmprr_pwΔmpφ::T = Δr_ext_inv_r_pwmprr * pwΔmpφ
            Δr_ext_inv_l_pwmprl_pwΔmpφ::T = Δr_ext_inv_l_pwmprl * pwΔmpφ
            pwΔmpφ_Δmpr_squared::T = pwΔmpφ * Δmpr_squared
            r_inv_pwΔmpr_Δφ_ext_inv_r::T = r_inv_pwΔmpr * Δφ_ext_inv_r
            r_inv_pwΔmpr_Δφ_ext_inv_l::T = r_inv_pwΔmpr * Δφ_ext_inv_l

            @fastmath @inbounds for iz in 2:(size(fssrb.potential, 1) - 1)
                inz::Int = nidx(iz, update_even_points, rφi_is_even)::Int
                # izr::Int = get_rbidx_right_neighbour(iz, update_even_points, rφi_is_even)::Int # this is somehow slower than the two lines below
                izr::Int = ifelse( rφi_is_even, iz, even_points ? iz - 1 : iz + 1)
                izr += ifelse(even_points, 1, 0)

                pwwzr::T        = gw_z[1, inz]
                pwwzl::T        = gw_z[2, inz]
                pwΔmpz::T       = gw_z[3, inz]
                Δz_ext_inv_r::T = gw_z[4, inz + 1]
                Δz_ext_inv_l::T = gw_z[4, inz]

                ϵ_rrr::T = fssrb.ϵ[  ir,  iφ, inz + 1]
                ϵ_rlr::T = fssrb.ϵ[  ir, inφ, inz + 1]
                ϵ_rrl::T = fssrb.ϵ[  ir,  iφ, inz ]
                ϵ_rll::T = fssrb.ϵ[  ir, inφ, inz ]
                ϵ_lrr::T = fssrb.ϵ[ inr,  iφ, inz + 1]
                ϵ_llr::T = fssrb.ϵ[ inr, inφ, inz + 1]
                ϵ_lrl::T = fssrb.ϵ[ inr,  iφ, inz ]
                ϵ_lll::T = fssrb.ϵ[ inr, inφ, inz ]

                vrr::T = fssrb.potential[     iz,     iφ, ir + 1, rb_src_idx]
                vrl::T = fssrb.potential[     iz,     iφ,    inr, rb_src_idx]
                vφr::T = fssrb.potential[     iz, iφ + 1,     ir, rb_src_idx]
                vφl::T = fssrb.potential[     iz,    inφ,     ir, rb_src_idx]
                vzr::T = fssrb.potential[    izr,     iφ,     ir, rb_src_idx]
                vzl::T = fssrb.potential[izr - 1,     iφ,     ir, rb_src_idx]

                pwwφr_pwwzr::T = pwwφr * pwwzr
                pwwφl_pwwzr::T = pwwφl * pwwzr
                pwwφr_pwwzl::T = pwwφr * pwwzl
                pwwφl_pwwzl::T = pwwφl * pwwzl
                pwwrl_pwwzr::T = pwwrl * pwwzr
                pwwrr_pwwzr::T = pwwrr * pwwzr
                pwwrl_pwwzl::T = pwwrl * pwwzl
                pwwrr_pwwzl::T = pwwrr * pwwzl

                # right weight in r: wrr
                wrr::T = ϵ_rrr * pwwφr_pwwzr
                wrr    = muladd(ϵ_rlr, pwwφl_pwwzr, wrr)
                wrr    = muladd(ϵ_rrl, pwwφr_pwwzl, wrr)
                wrr    = muladd(ϵ_rll, pwwφl_pwwzl, wrr)
                # left weight in r: wrr
                wrl::T = ϵ_lrr * pwwφr_pwwzr
                wrl    = muladd(ϵ_llr, pwwφl_pwwzr, wrl)
                wrl    = muladd(ϵ_lrl, pwwφr_pwwzl, wrl)
                wrl    = muladd(ϵ_lll, pwwφl_pwwzl, wrl)
                # right weight in φ: wφr
                wφr::T = ϵ_lrr * pwwrl_pwwzr
                wφr    = muladd(ϵ_rrr, pwwrr_pwwzr, wφr)
                wφr    = muladd(ϵ_lrl, pwwrl_pwwzl, wφr)
                wφr    = muladd(ϵ_rrl, pwwrr_pwwzl, wφr)
                # left weight in φ: wφl
                wφl::T = ϵ_llr * pwwrl_pwwzr
                wφl    = muladd(ϵ_rlr, pwwrr_pwwzr, wφl)
                wφl    = muladd(ϵ_lll, pwwrl_pwwzl, wφl)
                wφl    = muladd(ϵ_rll, pwwrr_pwwzl, wφl)
                # right weight in z: wzr
                wzr::T = ϵ_rrr * pwwrr_pwwφr
                wzr    = muladd(ϵ_rlr, pwwrr_pwwφl, wzr)
                wzr    = muladd(ϵ_lrr, pwwrl_pwwφr, wzr)
                wzr    = muladd(ϵ_llr, pwwrl_pwwφl, wzr)
                # left weight in z: wzr
                wzl::T = ϵ_rrl * pwwrr_pwwφr
                wzl    = muladd(ϵ_rll, pwwrr_pwwφl, wzl)
                wzl    = muladd(ϵ_lrl, pwwrl_pwwφr, wzl)
                wzl    = muladd(ϵ_lll, pwwrl_pwwφl, wzl)

                wrr *= Δr_ext_inv_r_pwmprr_pwΔmpφ * pwΔmpz
                wrl *= Δr_ext_inv_l_pwmprl_pwΔmpφ * pwΔmpz
                wφr *= r_inv_pwΔmpr_Δφ_ext_inv_r * pwΔmpz
                wφl *= r_inv_pwΔmpr_Δφ_ext_inv_l * pwΔmpz
                wzr *= Δz_ext_inv_r * pwΔmpφ_Δmpr_squared
                wzl *= Δz_ext_inv_l * pwΔmpφ_Δmpr_squared

                new_potential::T = _is_weighting_potential ? 0 : (fssrb.ρ[iz, iφ, ir, rb_tar_idx] + fssrb.ρ_fix[iz, iφ, ir, rb_tar_idx])
                new_potential = muladd( wrr, vrr, new_potential)
                new_potential = muladd( wrl, vrl, new_potential)
                new_potential = muladd( wφr, vφr, new_potential)
                new_potential = muladd( wφl, vφl, new_potential)
                new_potential = muladd( wzr, vzr, new_potential)
                new_potential = muladd( wzl, vzl, new_potential)

                new_potential *= fssrb.volume_weights[iz, iφ, ir, rb_tar_idx]

                old_potential::T = fssrb.potential[iz, iφ, ir, rb_tar_idx]

                new_potential -= old_potential
                new_potential = muladd(new_potential, fssrb.sor_const[inr], old_potential)

                if depletion_handling_enabled
                    if inr == 1 vrl = vrr end
                    if _bulk_is_ptype # p-type detectors
                        if new_potential < fssrb.minimum_applied_potential
                            # new_potential = fssrb.minimum_applied_potential
                            new_potential -= fssrb.ρ[iz, iφ, ir, rb_tar_idx] * fssrb.volume_weights[iz, iφ, ir, rb_tar_idx] * fssrb.sor_const[inr]
                            if (fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] & undepleted_bit == 0) fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] += undepleted_bit end # mark this point as undepleted
                        else
                            vmin::T = ifelse( vrr <  vrl, vrr,  vrl)
                            vmin    = ifelse( vφr < vmin, vφr, vmin)
                            vmin    = ifelse( vφl < vmin, vφl, vmin)
                            vmin    = ifelse( vzr < vmin, vzr, vmin)
                            vmin    = ifelse( vzl < vmin, vzl, vmin)
                            if new_potential <= vmin # bubble point
                                new_potential -= fssrb.ρ[iz, iφ, ir, rb_tar_idx] * fssrb.volume_weights[iz, iφ, ir, rb_tar_idx] * fssrb.sor_const[inr]
                                if (fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] & undepleted_bit == 0) fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] += undepleted_bit end # mark this point as undepleted
                            else # normal point
                                if (fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] & undepleted_bit > 0) fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] -= undepleted_bit end # unmark this point
                            end
                        end
                    else # n-type detectors
                        if new_potential > fssrb.maximum_applied_potential
                            # new_potential = fssrb.maximum_applied_potential
                            new_potential -= fssrb.ρ[iz, iφ, ir, rb_tar_idx] * fssrb.volume_weights[iz, iφ, ir, rb_tar_idx] * fssrb.sor_const[inr]
                            if (fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] & undepleted_bit == 0) fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] += undepleted_bit end # mark this point as undepleted
                        else
                            vmax::T = ifelse( vrr >  vrl, vrr,  vrl)
                            vmax    = ifelse( vφr > vmax, vφr, vmax)
                            vmax    = ifelse( vφl > vmax, vφl, vmax)
                            vmax    = ifelse( vzr > vmax, vzr, vmax)
                            vmax    = ifelse( vzl > vmax, vzl, vmax)
                            if new_potential >= vmax # bubble point
                                new_potential -= fssrb.ρ[iz, iφ, ir, rb_tar_idx] * fssrb.volume_weights[iz, iφ, ir, rb_tar_idx] * fssrb.sor_const[inr]
                                if (fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] & undepleted_bit == 0) fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] += undepleted_bit end # mark this point as undepleted
                            else # normal point -> unmark
                                if (fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] & undepleted_bit > 0) fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] -= undepleted_bit end # unmark this point
                            end
                        end
                    end
                end

                fssrb.potential[iz, iφ, ir, rb_tar_idx]::T = ifelse(fssrb.pointtypes[iz, iφ, ir, rb_tar_idx] & update_bit > 0, new_potential, old_potential)
            end # z loop
        end # φ loop
    end # inbounds
end


function update!(   fssrb; use_nthreads::Int = Base.Threads.nthreads(),
                    depletion_handling::Val{depletion_handling_enabled} = Val{false}(), only2d::Val{only_2d} = Val{false}(),
                    is_weighting_potential::Val{_is_weighting_potential} = Val{false}())::Nothing where {T, depletion_handling_enabled, only_2d, _is_weighting_potential}
    update!(fssrb, use_nthreads, Val{true}(), depletion_handling, Val{fssrb.bulk_is_ptype}(), is_weighting_potential)
    apply_boundary_conditions!(fssrb, Val{true}(), only2d)
    update!(fssrb, use_nthreads, Val{false}(), depletion_handling, Val{fssrb.bulk_is_ptype}(), is_weighting_potential)
    apply_boundary_conditions!(fssrb, Val{false}(), only2d)
    nothing
end

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

fssrb_ = SolidStateDetectors.PotentialSimulationSetupRB(detector, grid);
fssrb = (;[x=>getfield(fssrb_,x) for x in fieldnames(typeof(fssrb_)) if !in(x, (:grid, :geom_weights))]...);

gw1 = Float32[0.5 0.4375001 0.5333314 0.5000051 0.49999422 0.5000054 0.49999413 0.5000054 0.30434027 0.56250536 0.94117653; 0.5 0.56249994 0.46666864 0.49999487 0.5000058 0.49999455 0.50000584 0.49999455 0.69565976 0.43749467 0.058823477; 2.0 0.888889 0.46874812 0.33333492 0.24999994 0.20000046 0.16666666 0.14285737 0.089843825 0.05925864 0.53124994; 0.5 1.7857138 2.5000157 3.4999547 4.5000453 5.499932 6.5000677 7.499909 18.786152 15.500015 1.5; -0.0 0.5 1.7857138 2.5000157 3.4999547 4.5000453  5.499932 6.5000677 7.499909 18.786152 15.500015; 3.125e-6 2.0987658e-5 3.7615562e-5 5.925935e-5 7.9012425e-5 9.876555e-5 0.000118518714 0.00013827183 0.00011158403 8.364115e-5 0.0010492187]  # r or x
gw2 = Float32[0.5 0.88539994 0.5303906 0.5000026 0.5000002 0.49999574 0.50000405 0.5 0.49999565 0.33333653 0.49999997 0.49999997 0.0; 0.5 0.114600115 0.46960935 0.4999974 0.4999999 0.50000423 0.4999959 0.5 0.50000435 0.66666347 0.49999997 0.49999997 0.0; 0.00666716 0.029088803 0.05484392 0.058177702 0.058177993 0.058177516 0.05817747 0.058178008 0.0581775 0.043632954 0.029088855 0.029088974 0.0; 149.9889 149.9889 19.41354 17.188805 17.188631 17.188625 17.188921 17.188635 17.188625 17.188925 34.37739 34.377357 34.377357]  # φ or y
gw3 = Float32[0.090909086 0.33333334 0.59090835 0.5987666 0.04433465 0.95631045 0.52415663 0.039822955 0.95082045 0.4629611 0.37500083 0.6399998 0.5555549 0.90000004 0.0; 0.9090908 0.6666667 0.40909162 0.40123335 0.9556653 0.043689575 0.47584334 0.96017706 0.049179584 0.53703886 0.62499917 0.36000016 0.4444451 0.1 0.0; 0.055 0.0075 0.0061111 0.009 0.0056388993 0.0057222005 0.011500003 0.0062777996 0.005083345 0.009000003 0.0066666454 0.006944455 0.009999998 0.055555552 0.0; 10.0 100.0 200.0 138.46198 92.78331 2000.0034 91.370926 82.94899 2000.0109 103.44792 120.00049 199.9999 112.499855 90.00007 10.000001]  # z or z
fssrb = (;geom_weights=((weights=gw1,),(weights=gw2,),(weights=gw3,)), fssrb...,)

even_points = true
update_even_points = Val{even_points}()
depletion_handling = Val{false}()
rb_tar_idx, rb_src_idx = even_points ? (SSD.rb_even, SSD.rb_odd) : (SSD.rb_odd, SSD.rb_even)
# for idx3 in 2:(size(fssrb.potential, 3) - 1)
idx3 = 2
bulk_is_ptype = Val{fssrb.bulk_is_ptype}()
is_weighting_potential = Val{false}()

innerloops!(idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3,
            fssrb, update_even_points, depletion_handling,
            bulk_is_ptype, is_weighting_potential)
@info "Innerloop:"
@btime innerloops!(idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3,
            fssrb, update_even_points, depletion_handling,
            bulk_is_ptype, is_weighting_potential)




update!(fssrb, use_nthreads, update_even_points,
                            depletion_handling, bulk_is_ptype, is_weighting_potential)
@info "update!: $(use_nthreads) threads"
@btime update!(fssrb, use_nthreads, update_even_points,
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
