MODULE mo_interp
  implicit none
  private
  public :: interp, interp_fld, interp_multi_fld

contains

  SUBROUTINE interp(psrc,ptgt,tgtlevs,weights,nx,ny,nlevsrc,nlevtgt)
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
    use omp_lib

    integer(intk), intent(in), value  :: nx,ny
    integer(intk), intent(in), value  :: nlevsrc
    integer(intk), intent(in), value  :: nlevtgt

    real(dp), intent(in), dimension(nlevsrc,ny,nx) :: psrc
    real(dp), intent(in), dimension(nlevtgt,ny,nx) :: ptgt

    integer(intk), intent(out) :: tgtlevs(nlevtgt,ny,nx)
    real(dp),     intent(out) :: weights(nlevtgt,ny,nx)

    integer :: j,i,ksrc,ktgt, ii, ithr, nthr
    integer :: ksrcstart

    !write(*,*) "nx: ",nx," ny: ",ny," nlevsrc: ",nlevsrc," nlevtgt: ",nlevtgt
    !  write(*,'("Using ",I0," threads for dimension nx (",I0,")")') omp_get_num_threads(), nx
    !endif
    !$OMP PARALLEL DO PRIVATE(i, j, ktgt, ksrc, ksrcstart)
    do i=1,nx
      do j=1,ny
      associate(this_tgt => ptgt(:,j,i), this_src => psrc(:,j,i), &
          this_weights => weights(:,j,i), &
          this_tgtlevs => tgtlevs(:,j,i))

        ksrcstart=2
        do ktgt=1,nlevtgt
          ! No interpolation cases
          if (this_tgt(ktgt) <= this_src(1)) then
            this_tgtlevs(ktgt) = 1
            this_weights(ktgt) = 0

            cycle
          else if (this_tgt(ktgt) >= this_src(nlevsrc)) then
            this_tgtlevs(ktgt) = nlevsrc-1
            this_weights(ktgt) = 1
            cycle
          endif

        srcloop:  do ksrc=ksrcstart,nlevsrc

            if (this_tgt(ktgt) < this_src(ksrc)) then
                this_tgtlevs(ktgt) = (ksrc-1)
                this_weights(ktgt) = (this_tgt(ktgt) - this_src(ksrc-1))/ &
                                   & (this_src(ksrc) - this_src(ksrc-1))
                ksrcstart = ksrc
                exit srcloop
            endif

          end do srcloop
        end do
      end associate
      end do
    end do
    !$OMP END PARALLEL DO

!    write(*,*)  "OMP time: ",omp_get_wtime()-timer,"s"

  END SUBROUTINE interp

  SUBROUTINE interp_fld(fsrc, fdst, tgtlevs, weights, nx, ny, nlevsrc, nlevdst)
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
    use omp_lib

    integer(intk), intent(in), value  :: nx, ny
    integer(intk), intent(in), value  :: nlevdst, nlevsrc

    real(dp), intent(in),  dimension(nlevsrc,ny,nx) :: fsrc
    real(dp), intent(out), dimension(nlevdst,ny,nx) :: fdst

    integer(intk), intent(in) :: tgtlevs(nlevdst,ny,nx)
    real(dp),      intent(in) :: weights(nlevdst,ny,nx)

    integer :: i,j,k
    real(dp) :: timer

    timer = omp_get_wtime()
    !write(*,*) "nx: ",nx," ny: ",ny," nlevsrc: ",nlevsrc," nlevdst: ",nlevdst
    !if (omp_get_thread_num() == 0) then
    !  write(*,'("Using ",I0," threads for dimension nx (",I0,")")') omp_get_num_threads(), nx
    !endif
    !$OMP PARALLEL DO PRIVATE(i, j, k) SCHEDULE(DYNAMIC)
    do i=1,nx
      do j=1,ny
        associate(this_src => fsrc(:,j,i), this_dst => fdst(:,j,i), &
          this_tgtlevs => tgtlevs(:,j,i), &
          this_weights => weights(:,j,i))
            do k=1,nlevdst
              this_dst(k) = this_src(this_tgtlevs(k)+1)*this_weights(k) + &
                          & this_src(this_tgtlevs(k))*(1-this_weights(k))

            end do
        end associate
      end do
    end do

    !$OMP END PARALLEL DO
!    write(*,*)  "OMP time: ",omp_get_wtime()-timer,"s"
  END SUBROUTINE

  SUBROUTINE interp_multi_fld(fsrc, fdst, tgtlevs, weights, nflds, nx, ny, nlevsrc, nlevdst)
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
    use omp_lib

    integer(intk), intent(in), value  :: nflds
    integer(intk), intent(in), value  :: nx, ny
    integer(intk), intent(in), value  :: nlevdst, nlevsrc

    real(dp), intent(in),  dimension(nlevsrc,ny,nx,nflds) :: fsrc
    real(dp), intent(out), dimension(nlevdst,ny,nx,nflds) :: fdst

    integer(intk), intent(in) :: tgtlevs(nlevdst,ny,nx)
    real(dp),      intent(in) :: weights(nlevdst,ny,nx)

    integer :: f

    !write(*,*) "nx: ",nx," ny: ",ny," nlevsrc: ",nlevsrc," nlevdst: ",nlevdst
    !if (omp_get_thread_num() == 0) then
    !  write(*,'("Using ",I0," threads for (",I0,") fields")') omp_get_num_threads(), nflds
    !endif
    do f=1,nflds
        associate(this_src => fsrc(:,:,:,f), this_dst => fdst(:,:,:,f))
          call interp_fld(this_src, this_dst, tgtlevs, weights, nx, ny, nlevsrc, nlevdst)
        end associate
    end do
  END SUBROUTINE

  SUBROUTINE f_interp(psrc,ptgt,tgtlevs,weights,nx,ny,nlevsrc,nlevtgt) bind(C, name="interp")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value  :: nx,ny
    integer(intk), intent(in), value  :: nlevsrc
    integer(intk), intent(in), value  :: nlevtgt

    real(dp), intent(in), dimension(nlevsrc,ny,nx) :: psrc
    real(dp), intent(in), dimension(nlevtgt,ny,nx) :: ptgt

    integer(intk), intent(out) :: tgtlevs(nlevtgt,ny,nx)
    real(dp),      intent(out) :: weights(nlevtgt,ny,nx)

    call interp(psrc,ptgt,tgtlevs,weights,nx,ny,nlevsrc,nlevtgt)

  END SUBROUTINE f_interp

  SUBROUTINE f_interp_fld(fsrc, fdst, tgtlevs, weights, nx, ny, nlevsrc, nlevdst) bind(C,  name="interp_fld")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value  :: nx, ny
    integer(intk), intent(in), value  :: nlevsrc, nlevdst

    real(dp), intent(in),  dimension(nlevsrc,ny,nx) :: fsrc
    real(dp), intent(out), dimension(nlevdst,ny,nx) :: fdst

    integer(intk), intent(in) :: tgtlevs(nlevdst,ny,nx)
    real(dp),      intent(in) :: weights(nlevdst,ny,nx)

    call interp_fld(fsrc, fdst, tgtlevs, weights, nx, ny, nlevsrc, nlevdst)

  END SUBROUTINE f_interp_fld

  SUBROUTINE f_interp_multi_fld(fsrc, fdst, tgtlevs, weights, nflds, nx, ny, nlevsrc, nlevdst) bind(C,  name="interp_multi_fld")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value  :: nflds
    integer(intk), intent(in), value  :: nx, ny
    integer(intk), intent(in), value  :: nlevsrc, nlevdst

    real(dp), intent(in),  dimension(nlevsrc,ny,nx,nflds) :: fsrc
    real(dp), intent(out), dimension(nlevdst,ny,nx,nflds) :: fdst

    integer(intk), intent(in) :: tgtlevs(nlevdst,ny,nx)
    real(dp),      intent(in) :: weights(nlevdst,ny,nx)

    call interp_multi_fld(fsrc, fdst, tgtlevs, weights, nflds, nx, ny, nlevsrc, nlevdst)

  END SUBROUTINE f_interp_multi_fld
END MODULE mo_interp
