using Serialization, InteractiveUtils, BenchmarkTools

const rb_even = Int(2)
const rb_odd  = Int(1)

const rb_bool_even = true
const rb_bool_odd  = false
const update_bit   = 0x01

@inline function nidx( rbidx::Int, ::Val{true}, b::Bool)::Int
   return (rbidx - 1) * 2 - !b
end
@fastmath function update!( fssrb,
                            update_even_points::Val{even_points},
                            depletion_handling::Val{depletion_handling_enabled},
                            bulk_is_ptype::Val{_bulk_is_ptype},
                            is_weighting_potential::Val{_is_weighting_potential})::Nothing where {even_points, depletion_handling_enabled, _bulk_is_ptype, _is_weighting_potential}
    @inbounds begin
        rb_tar_idx::Int, rb_src_idx::Int = even_points ? (rb_even::Int, rb_odd::Int) : (rb_odd::Int,rb_even::Int)

        gw1 = fssrb.geom_weights[1].weights  # r or x
        gw2 = fssrb.geom_weights[2].weights  # φ or y
        gw3 = fssrb.geom_weights[3].weights  # z or z

        Base.Threads.@threads for idx3 in 2:(size(fssrb.potential, 3) - 1)
            innerloops!( idx3, rb_tar_idx, rb_src_idx, gw1, gw2, gw3, fssrb, update_even_points, depletion_handling, bulk_is_ptype, is_weighting_potential)
        end
    end
    nothing
end

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

fssrb = open(deserialize, "data.jls")

@info "$(VERSION): $(Threads.nthreads()) threads"
@btime update!(fssrb, Val{true}(), Val{false}(), Val{false}(), Val{false}())
