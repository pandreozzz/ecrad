PROGRAM testvertinterp
  use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
  use mo_interp, only : interp, interp_fld, interp_multi_fld

  integer(intk), parameter :: nflds=15
  integer(intk), parameter :: nx=1440, ny=720
  integer(intk), parameter :: nlevsrc=60, nlevtgt=137

  real(dp),      allocatable, dimension(:,:,:) :: p_tgt, weights
  integer(intk), allocatable, dimension(:,:,:) :: tgtlevs
  real(dp),      allocatable, dimension(:,:,:,:) :: p_src, p_dst

  integer :: i,j,k,f
  allocate(p_src(nlevsrc,ny,nx,nflds))
  allocate(p_tgt(nlevtgt,ny,nx))
  allocate(p_dst(nlevtgt,ny,nx,nflds))
  allocate(weights(nlevtgt,ny,nx))
  allocate(tgtlevs(nlevtgt,ny,nx))

  write(*,*) "Populating arrays..."
  do f=1,nflds
    do i=1,nx
      do j=1,ny
        do k=1,nlevsrc
          p_src(k,j,i,f) = i+ny*j+k**2+f*nlevsrc**2
        enddo
        do k=1,nlevtgt
          p_tgt(k,j,i) = i**2+ny+10*k+f*nlevtgt**2
        enddo
      enddo
    enddo
  enddo

  write(*,*) "Calling subroutine..."
  call interp(p_src(:,:,:,1), p_tgt, tgtlevs, weights, nx, ny, nlevsrc, nlevtgt)
  call interp_fld(p_src(:,:,:,1), p_dst(:,:,:,1), tgtlevs, weights, nx, ny, nlevsrc, nlevtgt)
  write(*,*) "Single field done. Calling multifield..."
  call interp_multi_fld(p_src, p_dst, tgtlevs, weights, nflds, nx, ny, nlevsrc, nlevtgt)

  do k=1,nlevtgt,10
    write(*,*) tgtlevs(k,1,1)
  end do
  deallocate(p_src)
  deallocate(p_tgt)
  deallocate(p_dst)
  deallocate(weights)
  deallocate(tgtlevs)
END PROGRAM
