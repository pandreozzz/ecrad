MODULE mo_lut_tools
  implicit none
  private
  public :: get_cdnc, get_bin_nearest

contains
  SUBROUTINE get_cdnc(species_mass, lut_map, mass_bins, cdnc_out, &
                    & nx, ny, nz, nspec, nbins, &
                    & lut_map_sep, ncon_bins, &
                    & bin_num, ccn_num, ccn_act)
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
    !use omp_lib

    integer(intk), intent(in), value :: nx, ny, nz
    integer(intk), intent(in), value :: nspec, nbins

    real(dp), intent(in), dimension(nspec,nz,ny,nx) :: species_mass
    real(dp), intent(in), dimension(nbins,nbins,nbins,nbins) :: lut_map
    real(dp), intent(in), dimension(nbins,nspec) :: mass_bins

    real(dp), intent(out), dimension(nz,ny,nx) :: cdnc_out

    ! Optional
    real(dp), intent(in), optional, dimension(nbins,nbins,nbins,nbins,nspec) :: lut_map_sep
    real(dp), intent(in), optional, dimension(nbins,nspec) :: ncon_bins
    real(dp), intent(out), optional, dimension(nspec,nz,ny,nx) :: ccn_num, ccn_act
    integer(intk), intent(out), optional, dimension(nspec,nz,ny,nx) :: bin_num

    integer(intk), dimension(nspec) :: tgtbins
    integer(intk) :: i,j,k,s

    logical :: with_sep_ccn = .false.

    if (present(lut_map_sep) .or. present(ncon_bins) .or. &
      & present(bin_num) .or. present(ccn_num) .or. present(ccn_act)) then
       if (present(lut_map_sep) .and. present(ncon_bins) .and. &
        & present(bin_num) .and. present(ccn_num) .and. present(ccn_act)) then
        with_sep_ccn = .true.
      else
        write(*,*) "Warning in get_cdnc: one among lut_map_sep, ncon_bins, bin_num, ccn_num, ccn_act is missing."
      endif
    endif


    !$OMP PARALLEL DO PRIVATE(j,k,s,tgtbins)
    do i=1,nx
      do j=1,ny
        do k=1,nz
        associate(this_mass => species_mass(:,k,j,i), this_cdnc => cdnc_out(k,j,i))

            call get_bin_nearest(this_mass, mass_bins, tgtbins, nspec, nbins)

            this_cdnc = lut_map(tgtbins(4), tgtbins(3), tgtbins(2), tgtbins(1))

            if (with_sep_ccn) then
              associate(this_bin_num => bin_num(:,k,j,i), &
                     & this_ccn_num => ccn_num(:,k,j,i),  &
                     & this_ccn_act => ccn_act(:,k,j,i))
                do s=1,nspec
                  this_ccn_num(s) = ncon_bins(tgtbins(s),s)
                  this_ccn_act(s) = lut_map_sep(tgtbins(4), tgtbins(3), tgtbins(2), tgtbins(1),s)/this_ccn_num(s)
                  this_bin_num(s) = tgtbins(s)
                enddo
              end associate
            endif

        end associate
        enddo
      enddo
    enddo
    !$OMP END PARALLEL DO

1000   END SUBROUTINE get_cdnc
  ! Reshape to allow flexible numbr of species


  SUBROUTINE get_bin_nearest(species_mass, mass_bins, tgtbins, nspec, nbins)
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32

    integer(intk), intent(in), value  :: nspec, nbins

    real(dp), intent(in), dimension(nspec) :: species_mass
    real(dp), intent(in), dimension(nbins,nspec) :: mass_bins

    integer(intk), intent(out), dimension(nspec) :: tgtbins

    real(dp) :: this_diff
    integer(intk) :: s
    integer(intk) :: b_lo,b_hi,b_mi

    do s=1,nspec
      associate(this_mass => species_mass(s), this_bins => mass_bins(:,s))
        b_lo = 1
        b_hi = nbins

        if (this_mass <= this_bins(b_lo)) then
          b_hi = b_lo
        else if (this_mass >= this_bins(b_hi)) then
          b_lo = b_hi
        endif

        ! Binary search
        do while ((b_hi - b_lo) > 1)
          b_mi = (b_hi + b_lo)/2
          this_diff = this_mass - this_bins(b_mi)

          if (this_diff >= 0) then
            b_lo = b_mi
          endif

          if (this_diff <= 0) then
            b_hi = b_mi
          endif
        enddo

        ! Choose nearest
        if (b_hi > b_lo) then
          if (2*this_mass-this_bins(b_lo)-this_bins(b_hi) < 0) then
            b_hi = b_lo
          else
            b_lo = b_hi
          endif
        endif

        tgtbins(s) = b_lo

      end associate
    enddo

  END SUBROUTINE get_bin_nearest

  SUBROUTINE f_get_cdnc(species_mass, lut_map, mass_bins, cdnc_out, &
                    & nx, ny, nz, nspec, nbins) bind(C,  name="get_cdnc")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value :: nx, ny, nz
    integer(intk), intent(in), value :: nspec, nbins

    real(dp), intent(in), dimension(nspec,nz,ny,nx) :: species_mass
    real(dp), intent(in), dimension(nbins,nbins,nbins,nbins) :: lut_map
    real(dp), intent(in), dimension(nbins,nspec) :: mass_bins

    real(dp), intent(out), dimension(nz,ny,nx) :: cdnc_out

    call get_cdnc(species_mass, lut_map, mass_bins, cdnc_out, &
                & nx, ny, nz, nspec, nbins)


   END SUBROUTINE f_get_cdnc

  SUBROUTINE f_get_cdnc_sep(species_mass, lut_map, mass_bins, cdnc_out, &
                    & nx, ny, nz, nspec, nbins, &
                    & lut_map_sep, ncon_bins, &
                    & bin_num, ccn_num, ccn_act) bind(C,  name="get_cdnc_sep")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value :: nx, ny, nz
    integer(intk), intent(in), value :: nspec, nbins

    real(dp), intent(in), dimension(nspec,nz,ny,nx) :: species_mass
    real(dp), intent(in), dimension(nbins,nbins,nbins,nbins) :: lut_map
    real(dp), intent(in), dimension(nbins,nspec) :: mass_bins

    real(dp), intent(out), dimension(nz,ny,nx) :: cdnc_out

    ! Optional
    real(dp), intent(in), optional, dimension(nbins,nbins,nbins,nbins,nspec) :: lut_map_sep
    real(dp), intent(in), optional, dimension(nbins,nspec) :: ncon_bins
    real(dp), intent(out), optional, dimension(nspec,nz,ny,nx) :: ccn_num, ccn_act
    integer(intk), intent(out), optional, dimension(nspec,nz,ny,nx) :: bin_num

    call get_cdnc(species_mass, lut_map, mass_bins, cdnc_out, &
                    & nx, ny, nz, nspec, nbins, &
                    & lut_map_sep, ncon_bins, &
                    & bin_num, ccn_num, ccn_act)

  END SUBROUTINE f_get_cdnc_sep

  SUBROUTINE f_get_bin_nearest(species_mass, mass_bins, tgtbins, nspec, nbins) bind(C, name="get_bin_nearest")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value  :: nspec, nbins

    real(dp), intent(in), dimension(nspec) :: species_mass
    real(dp), intent(in), dimension(nbins,nspec) :: mass_bins

    integer(intk), intent(out) :: tgtbins(nspec)

    real(dp) :: this_diff
    integer(intk) :: s
    integer(intk) :: b_lo,b_hi,b_mi

    call get_bin_nearest(species_mass, mass_bins, tgtbins, nspec, nbins)

  END SUBROUTINE f_get_bin_nearest
END MODULE mo_lut_tools
