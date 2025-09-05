MODULE mo_interp2d
  implicit none
  private
  public :: interp_2d

contains

  SUBROUTINE throw_error(error_string)
    character(*), intent(in) :: error_string
    write(*,*) error_string
  END SUBROUTINE

  FUNCTION get_dist(x1, x2) RESULT(dist)
  use, intrinsic :: iso_fortran_env, dp => real64, intk => int32

  real(dp), intent(in) :: x1, x2
  real(dp) :: dist

  dist = max(x1,x2) - min(x2,x1)
  if (dist > 180) then
    dist = min(x1,x2)+360 - max(x1,x2)
  endif  

  END FUNCTION

  FUNCTION get_geodist(dphi, phim, dlam) RESULT(geodist)
  use, intrinsic :: iso_fortran_env, dp => real64, intk => int32

  real(dp), intent(in) :: dphi, phim, dlam
  real(dp), parameter :: pi=acos(-1.0)
  real(dp) :: geodist

  geodist = sqrt((sin(dphi*pi/180)+cos(dlam*pi/180))**2 + (cos(phim*2*pi/180)*sin(dlam*pi/180)**2))
  
  END FUNCTION get_geodist

  SUBROUTINE get_interp_coeffs_reduced(ysrc, xsrc, ny_src, redpts_src, tgtx, tgty, xysrc, xyidxsrc, atol_in)
    use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32

    integer(intk), intent(in), value :: ny_src

    real(dp), intent(in), dimension(ny_src) :: ysrc
    real(dp), intent(in), dimension(ny_src) :: xsrc  
    integer(intk), intent(in), dimension(ny_src) :: redpts_src

    real(dp), intent(in) :: tgtx, tgty
    
    real(dp), intent(out), dimension(8) :: xysrc
    integer(intk), intent(out), dimension(4) :: xyidxsrc

    real(dp), intent(in), optional :: atol_in
    real(dp) :: atol = 1.e-6

    integer(intk) :: jlow, jmid, jhigh, ilow1, ihigh1, ilow2, ihigh2
    integer(intk) :: jj
    integer(intk) :: xidxsrc, yidxsrc
    logical :: foundx, foundy
    logical :: yisascending

    if (present(atol_in)) then
      atol = atol_in
    endif

    if (ysrc(ny_src) > ysrc(1)) then
      yisascending = .true.
    else
      yisascending = .false.
    endif

    foundy = .true.

    if (yisascending .and. (tgty < ysrc(1))) then
      jlow = 1
      jhigh = 1
    else if (yisascending .and. (tgty > ysrc(ny_src))) then
      jlow = ny_src
      jhigh = ny_src
    else if ((.not. yisascending) .and. (tgty < ysrc(ny_src))) then
      jlow = ny_src
      jhigh = ny_src
    else if ((.not. yisascending) .and. (tgty > ysrc(1))) then
      jlow = 1
      jhigh = 1
    else
      foundy = .false.
    endif

    if (.not. foundy) then
      !write(*,*) "Searching for y=",tgty
      if (yisascending) then
        jlow=1
        jhigh=ny_src
      else
        jlow=ny_src
        jhigh=1
      endif
      do while (abs(jhigh - jlow) > 1)
        jmid = (jlow + jhigh)/2
        if (abs(tgty - ysrc(jmid)) < atol) then
          jlow = jmid
          jhigh = jmid
        else if (tgty > ysrc(jmid)) then
          jlow = jmid
        else if (tgty < ysrc(jmid)) then
          jhigh = jmid
        endif
      end do
    endif

    !write(*,*) "Found y=",tgty," jlow=",jlow," and jhigh=",jhigh

    !xmax1 = 360 - 360./redpts_src(jlow) + xsrc(jlow)
    !xmax2 = 360 - 360./redpts_src(jhigh) + xsrc(jlow)
    ilow1 = mod(floor((tgtx-xsrc(jlow))/360.*redpts_src(jlow)),redpts_src(jlow))+1
    if (get_dist(tgtx, xsrc(jlow)+360/redpts_src(jlow)*(ilow1-1))<atol) then
      ihigh1 = ilow1
    else
      ihigh1 = mod(ilow1,redpts_src(jlow))+1
    endif
    ilow2 = mod(floor((tgtx-xsrc(jhigh))/360.*redpts_src(jhigh)),redpts_src(jhigh))+1
    if (get_dist(tgtx, xsrc(jhigh)+360/redpts_src(jhigh)*(ilow2-1))<atol) then
      ihigh2 = ilow2
    else
      ihigh2 = mod(ilow2,redpts_src(jhigh))+1
    endif

    ! A11 A12 A21 A22
    xyidxsrc(:) = 0
    do jj=1,jlow-1
      xyidxsrc(1) = xyidxsrc(1) + redpts_src(jj)
      xyidxsrc(2) = xyidxsrc(2) + redpts_src(jj)
    end do
    xyidxsrc(1) = xyidxsrc(1) + ilow1
    xyidxsrc(2) = xyidxsrc(2) + ihigh1
    do jj=1,jhigh-1
      xyidxsrc(3) = xyidxsrc(3) + redpts_src(jj)
      xyidxsrc(4) = xyidxsrc(4) + redpts_src(jj)
    end do
    xyidxsrc(3) = xyidxsrc(3) + ilow2
    xyidxsrc(4) = xyidxsrc(4) + ihigh2
    
    ! x11 x12 x21 x22 y11 y12 y21 y22
    xysrc(1) = xsrc(jlow) +360./redpts_src(jlow )*(ilow1 -1)
    xysrc(2) = xsrc(jlow) +360./redpts_src(jlow )*(ihigh1-1)
    xysrc(3) = xsrc(jhigh)+360./redpts_src(jhigh)*(ilow2 -1)
    xysrc(4) = xsrc(jhigh)+360./redpts_src(jhigh)*(ihigh2-1)
                
    xysrc(5) = ysrc(jlow)
    xysrc(6) = ysrc(jlow)
    xysrc(7) = ysrc(jhigh)
    xysrc(8) = ysrc(jhigh)

  END SUBROUTINE get_interp_coeffs_reduced

  SUBROUTINE get_interp_coeffs_lonlat(xsrc, ysrc, nx_src, ny_src, &
                                    & tgtx, tgty, periodic_domain, &
                                    & xysrc, xyidxsrc, atol_in)
    use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32

    integer(intk), intent(in), value :: ny_src, nx_src

    real(dp), intent(in), dimension(nx_src) :: xsrc  
    real(dp), intent(in), dimension(ny_src) :: ysrc

    real(dp), intent(in) :: tgtx, tgty
    real(dp), intent(out), dimension(8) :: xysrc
    integer(intk), intent(out), dimension(4) :: xyidxsrc

    logical, intent(in) :: periodic_domain

    real(dp), intent(in), optional :: atol_in
    real(dp) :: atol = 1.e-6

    integer(intk) :: imid, ilow, ihigh, jmid, jlow, jhigh
    integer(intk) :: xidxsrc, yidxsrc
    logical :: foundx, foundy
    logical :: xisascending, yisascending

    if (xsrc(nx_src) > xsrc(1)) then
      xisascending = .true.
    else
      xisascending = .false.
    endif

    if (ysrc(ny_src) > ysrc(1)) then
      yisascending = .true.
    else
      yisascending = .false.
    endif

    if (present(atol_in)) then
      atol = atol_in
    endif


    foundx = .true.
    foundy = .true.

    ! Handle boundaries
    if (xisascending .and. ((tgtx < xsrc(1)) .or. (tgtx > xsrc(nx_src)))) then
      ilow = nx_src
      ihigh = 1
    else if ((.not. xisascending) .and. ((tgtx < xsrc(nx_src)) .or. (tgtx > xsrc(1)))) then
      ilow = 1
      ihigh = nx_src
    else
      foundx = .false.
    endif

    if (yisascending .and. (tgty < ysrc(1))) then
      jlow = 1
      jhigh = 1
    else if (yisascending .and. (tgty > ysrc(ny_src))) then
      jlow = ny_src
      jhigh = ny_src
    else if ((.not. yisascending) .and. (tgty < ysrc(ny_src))) then
      jlow = ny_src
      jhigh = ny_src
    else if ((.not. yisascending) .and. (tgty > ysrc(1))) then
      jlow = 1
      jhigh = 1
    else
      foundy = .false.
    endif

    ! Binary search inside domain

    if (.not. foundx) then
      if (xisascending) then
        ilow=1
        ihigh=nx_src
      else
        ilow=nx_src
        ihigh=1
      endif
      do while (abs(ihigh - ilow) > 1)
        imid = (ilow + ihigh)/2

        if (abs(tgtx - xsrc(imid)) < atol) then
          ilow = imid
          ihigh = imid
        else if (tgtx > xsrc(imid)) then
          ilow = imid
        else if (tgtx < xsrc(imid)) then
          ihigh = imid
        endif
      end do
    endif

    if (.not. foundy) then
      if (yisascending) then
        jlow=1
        jhigh=ny_src
      else
        jlow=ny_src
        jhigh=1
      endif
      do while (abs(jhigh - jlow) > 1)
        jmid = (jlow + jhigh)/2
        if (abs(tgty - ysrc(jmid)) < atol) then
          jlow = jmid
          jhigh = jmid
        else if (tgty > ysrc(jmid)) then
          jlow = jmid
        else if (tgty < ysrc(jmid)) then
          jhigh = jmid
        endif
      end do
    endif
    

  !write(*,*) "ilow=",ilow," ihigh=",ihigh
  !write(*,*) "jlow=",jlow," jhigh=",jhigh
  !yx 11
  xyidxsrc(1) = (jlow-1)*nx_src+ilow
  ! yx 12
  xyidxsrc(2) = (jlow-1)*nx_src+ihigh
  ! yx 21
  xyidxsrc(3) = (jhigh-1)*nx_src+ilow
  ! yx 22
  xyidxsrc(4) = (jhigh-1)*nx_src+ihigh

  ! indices are (y,x)
  ! A11 A12 A21 A22
  !xysrc(1:4) = (/xsrc(ilow), xsrc(ihigh), xsrc(ilow), xsrc(ihigh)/)
  !xysrc(5:8) = (/ysrc(jlow), ysrc(jlow), ysrc(jhigh), ysrc(jhigh)/)
  xysrc(1) = xsrc(ilow)
  xysrc(2) = xsrc(ihigh)
  xysrc(3) = xsrc(ilow)
  xysrc(4) = xsrc(ihigh)
  xysrc(5) = ysrc(jlow)
  xysrc(6) = ysrc(jlow)
  xysrc(7) = ysrc(jhigh)
  xysrc(8) = ysrc(jhigh)

  END SUBROUTINE get_interp_coeffs_lonlat

  SUBROUTINE interp_2d(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                     & nsrc, &
                     & ny_src, nx_src, nxy_src, &
                     & ny_dst, nx_dst, nxy_dst, &
                     & typ_src, typ_dst, &
                     & redpts_src, redpts_dst, &
                     & chunk_size_in, abs_tolerance_in)

    use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
    use omp_lib
    
    ! Number of fields to interpolate
    integer(intk), intent(in), value  :: nsrc

    ! Grids shapes
    integer(intk), intent(in), value  :: ny_src, nx_src, nxy_src
    integer(intk), intent(in), value  :: ny_dst, nx_dst, nxy_dst
    
    ! Grids types
    integer(intk), intent(in), value  :: typ_src, typ_dst

    ! Fields source and interpolated
    real(dp), intent(in), dimension(nxy_src,nsrc) :: fsrc
    real(dp), intent(out), dimension(nxy_dst,nsrc) :: fdst
    
    ! Grid points
    real(dp), intent(in), dimension(nx_src) :: xsrc, ysrc
    real(dp), intent(in), dimension(nx_dst) :: xdst, ydst
    
    ! Only for reduced grids - x_ indicates the first longitude for each y_
    ! and redpts_ defines how many longitude points between x_ and 360. 
    ! per each y_. x_ <= 360./redpts_
    integer(intk), optional, intent(in), dimension(ny_src)  :: redpts_src
    integer(intk), optional, intent(in), dimension(ny_dst)  :: redpts_dst

    ! Specify chunk size
    integer(intk), intent(in), value, optional :: chunk_size_in
    ! Specify absolute tolerance (default is 10^-3 degrees)
    real(dp), intent(in), value, optional :: abs_tolerance_in
    

    ! Default parameters for chunk size and tolerance
    integer(intk) :: max_chunk_size = 1000
    integer(intk) :: actual_chunk_size
    real(dp) :: atol = 1.e-3

    logical :: yisascending

    ! variables for OMP
    integer(intk) :: num_threads, thread_id
    integer(intk) :: ndimomp, n_chunks, nc 
    ! loop variables
    integer(intk) :: jdst, jdst_min, jdst_max, idst, idst_min, idst_max, jj
    
    real(dp) :: tgtx, tgty
    integer(intk) :: tgtidx
    real(dp), dimension(8) :: xysrc
    integer(intk), dimension(4) :: xyidxsrc

    real(dp) :: x11,x12,x21,x22
    real(dp) :: y11,y12,y21,y22
    integer(intk) :: xyidxsrc11, xyidxsrc12, xyidxsrc21, xyidxsrc22 

    ! To temporarily store intermediate interpolations
    real(dp) :: xa, xb, wa, wb, w
    real(dp), dimension(nsrc) :: fxay, fxby

    real(dp) :: timer
    
    ! typ_ = 
    ! 1 = lonlat
    ! 2 = reduced
    ! 3 = unstructured

    
    ! rectangular
    ! present(redpts_) is false
    ! nxy_ = nx*ny
 
    ! unstructured
    ! present(redpts_) is false
    !
    ! nx_ = ny_ = nxy_
    ! case(3) assume ordered by lat,lon
    ! case(4) completely unstructured

    ! reduced gaussian
    ! present(redpts_) is true
    ! nx_ = ny_ != nxy_
    ! x_ indicates the first longitude for each y_
    select case(typ_src)
      case(1)
        if (nxy_src /= (nx_src*ny_src)) then
          call throw_error("nxy = nx*ny needed for rectangular grids!")
        endif
        write(*,*) "Src grid is rectangular"
      case(2)
        if ((nx_src /= ny_src) .or. (.not. present(redpts_src))) then
          call throw_error("nx_src must be equal to ny_src "&
                       & //"redpts_src must be provided for reduced grids!")
        endif
        write(*,*) "Src grid is reduced"
      case(3)
        if ((nx_src /= ny_src) .or. (ny_src/= nxy_src)) then
          call throw_error("nx_src = ny_src = nxy_src required for unstructured grids!")
        endif
        write(*,*) "Src grid is unstructured"
      case default
        call throw_error("Could not recognize typ_src")
    end select

    select case(typ_dst)
      case(1)
        if (nxy_dst /= (nx_dst*ny_dst)) then
          call throw_error("nxy = nx*ny needed for rectangular grids!")
        endif
        write(*,*) "tgt grid is rectangular"
      case(2)
        if ((nx_dst /= ny_dst) .or. (.not. present(redpts_dst))) then
          call throw_error("nx_dst must have the same length of ny_dst "&
                       & //"redpts_dst must be provided for reduced grids!")
        endif
        write(*,*) "tgt grid is reduced"
      case(3)
      case(4)
        if ((nx_dst /= ny_dst) .or. (ny_dst/= nxy_dst)) then
          call throw_error("nx_dst = ny_dst = nxy_dst required for unstructured grids!")
        endif
        write(*,*) "tgt grid is unstructured"
      case default
        call throw_error("Could not recognize typ_dst")
    end select

    timer = omp_get_wtime()

    if (present(abs_tolerance_in)) then
      atol = abs_tolerance_in
    endif

    !$OMP PARALLEL PRIVATE(thread_id, nc, jdst_min, jdst_max, jdst, idst_min, idst_max, idst, &
    !$OMP tgty, tgtx, tgtidx, xysrc, xyidxsrc, &
    !$OMP xyidxsrc11, xyidxsrc12, xyidxsrc21, xyidxsrc22, &
    !$OMP x11, x12, x21, x22, y11, y12, y21, y22, &
    !$OMP w, xa, fxay, xb, fxby)

    thread_id = int(omp_get_thread_num(), intk)

    ! Master thread decides
    if (thread_id == 0) then
        num_threads = int(omp_get_num_threads(), intk)
        write(*,'(A,I0,A)') "Interp_2d using ",num_threads," threads"

        ! Parallelize on y
        if (nxy_dst > num_threads) then
            ndimomp = ny_dst
        else
            ndimomp = 1
        endif

        ! Compute chunk size
        if (present(chunk_size_in)) then
          max_chunk_size = chunk_size_in
        elseif (typ_dst == 2) then
          ! This is because the workload for latitude ydst(j) 
          ! is determined by redpts_dst(j)
          max_chunk_size = 1
        endif
        actual_chunk_size = min(ceiling(ndimomp * 1.0/num_threads), max_chunk_size)
        n_chunks = ceiling(ndimomp * 1.0/actual_chunk_size)
    end if

    !$OMP BARRIER

    ! Initialize for each potentially parallelizable dimension
    do nc=1,n_chunks
      !is this my chunk?
      if ((mod(nc-1,num_threads)) /= thread_id) then
        cycle
      endif
      jdst_min=(nc-1)*actual_chunk_size+1
      jdst_max = min(ndimomp,jdst_min+actual_chunk_size-1)

      !write(*,*) "Thread ",thread_id," doing chunk ",nc,"/",n_chunks," from ",jdst_min," to ",jdst_max
      ! Along y (lats)        
      do jdst=jdst_min,jdst_max
        ! Find x range
        select case(typ_dst)
          case (1)
            idst_min=1
            idst_max=nx_dst
          case(2)
            idst_min=1
            idst_max=redpts_dst(jdst)
          case(3)
          case(4)
            idst_min=jdst
            idst_max=jdst
        end select
        tgty = ydst(jdst)
        !write(*,'(A,I0,A,I0,A,F6.2)') "Thread ",thread_id," chunk ",nc," tgty=",tgty
        do idst=idst_min,idst_max
          !write(*,*) "Thread ",thread_id," (jdst,idst)=(",jdst,",",idst,")"
          select case(typ_dst)
            case (1)
              tgtidx = (jdst-1)*nx_dst + idst 
              tgtx = xdst(idst)
            case (2)
              tgtidx = 0
                do jj=1,jdst-1
                  tgtidx = tgtidx + redpts_dst(jj)
                end do
              tgtidx = tgtidx + idst
              tgtx = xdst(jdst) + (idst - 1)*360.0/idst_max
            case (3)
            case (4)
              tgtidx = jdst
              tgtx = xdst(idst)
          end select
          !write(*,*) "Thread ",thread_id," (tgty,tgtx)=(",tgty,",",tgtx,")"

          select case(typ_src)
            case (1)
              !write(*,*) "Call interp coeffs lonlat"
              call get_interp_coeffs_lonlat(xsrc, ysrc, nx_src, ny_src, tgtx, tgty, .true., xysrc, xyidxsrc, atol_in=atol)
            case (2)
              !write(*,*) "Call interp coeffs reduced"
              call get_interp_coeffs_reduced(ysrc, xsrc, ny_src, redpts_src, &
              & tgtx, tgty, xysrc, xyidxsrc, atol_in=atol)
            !case (3)
            !  call get_interp_coeffs_unstruct(xsrc, ysrc, tgtx, tgty, xysrc(1:8), xyidxsrc(1:4), .true.)
            !case (4)
            !  call get_interp_coeffs_unstruct(xsrc, ysrc, tgtx, tgty, xysrc(1:8), xyidxsrc(1:4), .false.)
          end select

          ! indices are (y,x)
          ! A11 A12 A21 A22
          x11 = xysrc(1)
          x12 = xysrc(2)
          x21 = xysrc(3)
          x22 = xysrc(4)

          y11 = xysrc(5)
          y12 = xysrc(6)
          y21 = xysrc(7)
          y22 = xysrc(8)

          xyidxsrc11 = xyidxsrc(1)
          xyidxsrc12 = xyidxsrc(2)
          xyidxsrc21 = xyidxsrc(3)
          xyidxsrc22 = xyidxsrc(4)


          !                                       . A22(y22,x22)
          !      . A21(y21,x21)
          !   . Aay(tgty,xa)      . (tgty,tgtx)    . Aby(tgty,xb)
          ! . A11(y11,x11)
          !
          !                                          . A12(y12,x12)

          !write(*,'(A,I0,A,F6.2,A,F6.2)') "Thread ",thread_id,"tgty=",tgty," tgtx=",tgtx
                 
          ! Find xb and interpolate right side
          if ((get_dist(x22,x12)>atol) .and. (y22-y12>atol)) then
            w = (y22-tgty)/(y22-y12)
            if (x22-x12>180) then
              xb = w*(x12+360)+(1-w)*x22
            else if (x22-x12<-180) then
              xb = w*x12+(1-w)*(x22+360)
            else
              xb = w*x12+(1-w)*x22
            endif
            
            if (xb > 360) then
              xb = modulo(xb,360.)
            endif
          else
            xb = x12
          endif

          if ((get_dist(xb,x12)>atol)) then
            wb = sqrt((get_dist(xb,x12)**2 + (tgty-y12)**2)/(get_dist(x22,x12)**2+(y22-y12)**2))
            !wb = get_geodist((tgty-y12), (tgty+y12)/2., get_dist(xb,x12))/get_geodist((y22-y12), (y22+y12)/2., get_dist(x22,x12))
          else if (y22-y12>atol) then
            wb = (tgty-y12)/(y22-y12)
          else
            wb = 0.
          endif
          fxby(:) = wa*fsrc(xyidxsrc22,:) + (1-wa)*fsrc(xyidxsrc12,:)

          ! Find xa and interpolate left side
          if ((get_dist(x21,x11)>atol) .and. (y21-y11>atol)) then
            w = (y21-tgty)/(y21-y11)
            if (x21-x11>180) then
              xa = w*(x11+360)+(1-w)*x21
            else if (x21-x11<-180) then
              xa = w*x11+(1-w)*(x21+360)
            else
              xa = w*x11+(1-w)*x21
            endif
            if (xa > 360) then
              xa = modulo(xa,360.)
            endif
          else
            xa = x11
          endif
          if (get_dist(xa, x11)>atol) then
            wa = sqrt((get_dist(xa,x11)**2 + (tgty-y11)**2)/(get_dist(x21,x11)**2+(y21-y11)**2))
            !wa = get_geodist((tgty-y11), (tgty+y11)/2., get_dist(xa,x11))/get_geodist((y21-y11), (y21+y11)/2., get_dist(x21,x11))
          else if (y21-y11>atol) then
            wa = (tgty-y11)/(y21-y11)
          else
            wa = 0.
          endif
          fxay(:) = wa*fsrc(xyidxsrc21,:) + (1-wa)*fsrc(xyidxsrc11,:)
          
          if (get_dist(xb,xa)>atol) then
            w = get_dist(tgtx,xb)/get_dist(xb,xa)
            fdst(tgtidx,:) = w*fxay(:) + (1-w)*fxby(:)
          else
            fdst(tgtidx,:) = fxay(:)
          endif
          !write(*,'(A,F6.2,A,F6.2,A,F6.2,A,F6.2,A,'//&
          !&'F6.2,A,F6.2,A,F6.2,A,F6.2,A,F6.2,A,F6.2,'//&
          !&'A,I0,A,I0,A,I0,A,I0,A,F6.2,A,F6.2,A,F6.2,A,F6.2,A,F6.2,A)') &
          !&"tgty=",tgty," tgtx=",tgtx," "//new_line('a')//&
          !&"A21(",y21,",",x21,") A22(",y22,",",x22,") "//new_line('a')//&
          !&"A11(",y11,",",x11,") A12(",y12,",",x12,") "//new_line('a')//&
          !&"idx21=",xyidxsrc21," idx22=",xyidxsrc22," "//new_line('a')//&
          !&"idx11=",xyidxsrc11," idx12=",xyidxsrc12," "//new_line('a')//&
          !&"xa=",xa," xb=",xb," wa=",wa," wb=",wb," w=",w," "
          !write(*,'(A,I0,A,F6.2,A,F6.2,A,F6.2,A)') &
          !&"tgtidx=",tgtidx," fxay=",fxay(1),"  fxby=",fxby(1),&
          !&" fdst=",fdst(tgtidx,1)," "//new_line('a')
        end do
      end do
      !write(*,'(A,I0,A,I0,A)') "Thread ",thread_id," chunk ",nc," finished"
    end do
    !$OMP END PARALLEL

  write(*,*)  "OMP time: ",omp_get_wtime()-timer,"s"

  END SUBROUTINE interp_2d


  SUBROUTINE f_interp_2d_rec2rec(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                               & nsrc, &
                               & ny_src, nx_src, nxy_src, &
                               & ny_dst, nx_dst, nxy_dst, &
                               & chunk_size_in, abs_tolerance_in) &
            & bind(C, name="interp_2d_rec2rec")
    use iso_c_binding, dp => c_double, intk => c_int32_t
    
    ! Number of fields to interpolate
    integer(intk), intent(in), value  :: nsrc

    ! Grids shapes
    integer(intk), intent(in), value  :: ny_src, nx_src, nxy_src
    integer(intk), intent(in), value  :: ny_dst, nx_dst, nxy_dst
    

    ! Fields source and interpolated
    real(dp), intent(in), dimension(nxy_src,nsrc) :: fsrc
    real(dp), intent(out), dimension(nxy_dst,nsrc) :: fdst
    
    ! Grid points
    real(dp), intent(in), dimension(nx_src) :: xsrc, ysrc
    real(dp), intent(in), dimension(nx_dst) :: xdst, ydst
    
    ! Only for reduced grids - x_ indicates the first longitude for each y_
    ! and redpts_ defines how many longitude points between x_ and 360. 
    ! per each y_. x_ <= 360./redpts_

    ! Specify chunk size
    integer(intk), intent(in), value :: chunk_size_in
    ! Specify absolute tolerance (default is 10^-3 degrees)
    real(dp), intent(in), value :: abs_tolerance_in
    
    ! Grids types
    integer(intk) :: typ_src=1
    integer(intk) :: typ_dst=1

    call interp_2d(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                 & nsrc, &
                 & ny_src, nx_src, nxy_src, &
                 & ny_dst, nx_dst, nxy_dst, &
                 & typ_src, typ_dst, &
                 & chunk_size_in=chunk_size_in, abs_tolerance_in=abs_tolerance_in)
  
  END SUBROUTINE f_interp_2d_rec2rec

  SUBROUTINE f_interp_2d_rec2red(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                               & nsrc, &
                               & ny_src, nx_src, nxy_src, &
                               & ny_dst, nx_dst, nxy_dst, &
                               & redpts_dst, &
                               & chunk_size_in, abs_tolerance_in) &
            & bind(C, name="interp_2d_rec2red")
    use iso_c_binding, dp => c_double, intk => c_int32_t
    
    ! Number of fields to interpolate
    integer(intk), intent(in), value  :: nsrc

    ! Grids shapes
    integer(intk), intent(in), value  :: ny_src, nx_src, nxy_src
    integer(intk), intent(in), value  :: ny_dst, nx_dst, nxy_dst
    

    ! Fields source and interpolated
    real(dp), intent(in), dimension(nxy_src,nsrc) :: fsrc
    real(dp), intent(out), dimension(nxy_dst,nsrc) :: fdst
    
    ! Grid points
    real(dp), intent(in), dimension(nx_src) :: xsrc, ysrc
    real(dp), intent(in), dimension(nx_dst) :: xdst, ydst
    
    ! Only for reduced grids - x_ indicates the first longitude for each y_
    ! and redpts_ defines how many longitude points between x_ and 360. 
    ! per each y_. x_ <= 360./redpts_
    integer(intk), intent(in), dimension(ny_dst)  :: redpts_dst

    ! Specify chunk size
    integer(intk), intent(in), value :: chunk_size_in
    ! Specify absolute tolerance (default is 10^-3 degrees)
    real(dp), intent(in), value :: abs_tolerance_in
    
    ! Grids types
    integer(intk) :: typ_src=1
    integer(intk) :: typ_dst=2

    call interp_2d(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                 & nsrc, &
                 & ny_src, nx_src, nxy_src, &
                 & ny_dst, nx_dst, nxy_dst, &
                 & typ_src, typ_dst, &
                 & redpts_dst=redpts_dst, &
                 & chunk_size_in=chunk_size_in, abs_tolerance_in=abs_tolerance_in)
  
  END SUBROUTINE f_interp_2d_rec2red
  
  SUBROUTINE f_interp_2d_red2rec(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                               & nsrc, &
                               & ny_src, nx_src, nxy_src, &
                               & ny_dst, nx_dst, nxy_dst, &
                               & redpts_src, &
                               & chunk_size_in, abs_tolerance_in) &
            & bind(C, name="interp_2d_red2rec")
    use iso_c_binding, dp => c_double, intk => c_int32_t
    
    ! Number of fields to interpolate
    integer(intk), intent(in), value  :: nsrc

    ! Grids shapes
    integer(intk), intent(in), value  :: ny_src, nx_src, nxy_src
    integer(intk), intent(in), value  :: ny_dst, nx_dst, nxy_dst
    

    ! Fields source and interpolated
    real(dp), intent(in), dimension(nxy_src,nsrc) :: fsrc
    real(dp), intent(out), dimension(nxy_dst,nsrc) :: fdst
    
    ! Grid points
    real(dp), intent(in), dimension(nx_src) :: xsrc, ysrc
    real(dp), intent(in), dimension(nx_dst) :: xdst, ydst
    
    ! Only for reduced grids - x_ indicates the first longitude for each y_
    ! and redpts_ defines how many longitude points between x_ and 360. 
    ! per each y_. x_ <= 360./redpts_
    integer(intk), intent(in), dimension(ny_src)  :: redpts_src

    ! Specify chunk size
    integer(intk), intent(in), value :: chunk_size_in
    ! Specify absolute tolerance (default is 10^-3 degrees)
    real(dp), intent(in), value :: abs_tolerance_in
    
    ! Grids types
    integer(intk) :: typ_src=2
    integer(intk) :: typ_dst=1

    call interp_2d(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                 & nsrc, &
                 & ny_src, nx_src, nxy_src, &
                 & ny_dst, nx_dst, nxy_dst, &
                 & typ_src, typ_dst, &
                 & redpts_src=redpts_src, &
                 & chunk_size_in=chunk_size_in, abs_tolerance_in=abs_tolerance_in)
  
  END SUBROUTINE f_interp_2d_red2rec

  SUBROUTINE f_interp_2d_red2red(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                               & nsrc, &
                               & ny_src, nx_src, nxy_src, &
                               & ny_dst, nx_dst, nxy_dst, &
                               & redpts_src, redpts_dst, &
                               & chunk_size_in, abs_tolerance_in) &
            & bind(C, name="interp_2d_red2red")
    use iso_c_binding, dp => c_double, intk => c_int32_t
    
    ! Number of fields to interpolate
    integer(intk), intent(in), value  :: nsrc

    ! Grids shapes
    integer(intk), intent(in), value  :: ny_src, nx_src, nxy_src
    integer(intk), intent(in), value  :: ny_dst, nx_dst, nxy_dst
    

    ! Fields source and interpolated
    real(dp), intent(in), dimension(nxy_src,nsrc) :: fsrc
    real(dp), intent(out), dimension(nxy_dst,nsrc) :: fdst
    
    ! Grid points
    real(dp), intent(in), dimension(nx_src) :: xsrc, ysrc
    real(dp), intent(in), dimension(nx_dst) :: xdst, ydst
    
    ! Only for reduced grids - x_ indicates the first longitude for each y_
    ! and redpts_ defines how many longitude points between x_ and 360. 
    ! per each y_. x_ <= 360./redpts_
    integer(intk), intent(in), dimension(ny_src)  :: redpts_src
    integer(intk), intent(in), dimension(ny_dst)  :: redpts_dst

    ! Specify chunk size
    integer(intk), intent(in), value :: chunk_size_in
    ! Specify absolute tolerance (default is 10^-3 degrees)
    real(dp), intent(in), value :: abs_tolerance_in
    
    ! Grids types
    integer(intk) :: typ_src=2
    integer(intk) :: typ_dst=2

    call interp_2d(fsrc, fdst, ysrc, xsrc, ydst, xdst, &
                  & nsrc, &
                  & ny_src, nx_src, nxy_src, &
                  & ny_dst, nx_dst, nxy_dst, &
                  & typ_src, typ_dst, &
                  & redpts_src=redpts_src, redpts_dst=redpts_dst, &
                  & chunk_size_in=chunk_size_in, abs_tolerance_in=abs_tolerance_in)
  
  END SUBROUTINE f_interp_2d_red2red

END MODULE mo_interp2d
