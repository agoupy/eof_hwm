!     -*- f90 -*-
!     This file is autogenerated with f2py (version:2)
!     It contains Fortran 90 wrappers to fortran functions.

      subroutine f2py_hwm_getdims_gpbar(r,s,f2pysetdata,flag)
      use hwm, only: d => gpbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_hwm_getdims_gpbar
      subroutine f2py_hwm_getdims_spbar(r,s,f2pysetdata,flag)
      use hwm, only: d => spbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_hwm_getdims_spbar
      subroutine f2py_hwm_getdims_gwbar(r,s,f2pysetdata,flag)
      use hwm, only: d => gwbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_hwm_getdims_gwbar
      subroutine f2py_hwm_getdims_gvbar(r,s,f2pysetdata,flag)
      use hwm, only: d => gvbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_hwm_getdims_gvbar
      subroutine f2py_hwm_getdims_svbar(r,s,f2pysetdata,flag)
      use hwm, only: d => svbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_hwm_getdims_svbar
      subroutine f2py_hwm_getdims_swbar(r,s,f2pysetdata,flag)
      use hwm, only: d => swbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_hwm_getdims_swbar
      
      subroutine f2pyinithwm(f2pysetupfunc)
      use hwm, only : nmaxgeo
      use hwm, only : gpbar
      use hwm, only : nmaxdwm
      use hwm, only : spbar
      use hwm, only : gwbar
      use hwm, only : glatalf
      use hwm, only : gvbar
      use hwm, only : nmaxqdc
      use hwm, only : omaxhwm
      use hwm, only : mmaxgeo
      use hwm, only : mmaxdwm
      use hwm, only : hwminit
      use hwm, only : svbar
      use hwm, only : mmaxqdc
      use hwm, only : nmaxhwm
      use hwm, only : swbar
      external f2pysetupfunc
      external f2py_hwm_getdims_gpbar
      external f2py_hwm_getdims_spbar
      external f2py_hwm_getdims_gwbar
      external f2py_hwm_getdims_gvbar
      external f2py_hwm_getdims_svbar
      external f2py_hwm_getdims_swbar
      call f2pysetupfunc(nmaxgeo,f2py_hwm_getdims_gpbar,nmaxdwm,f2py_hwm&
     &_getdims_spbar,f2py_hwm_getdims_gwbar,glatalf,f2py_hwm_getdims_gvb&
     &ar,nmaxqdc,omaxhwm,mmaxgeo,mmaxdwm,hwminit,f2py_hwm_getdims_svbar,&
     &mmaxqdc,nmaxhwm,f2py_hwm_getdims_swbar)
      end subroutine f2pyinithwm

      subroutine f2py_alf_getdims_en(r,s,f2pysetdata,flag)
      use alf, only: d => en

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_alf_getdims_en
      subroutine f2py_alf_getdims_bnm(r,s,f2pysetdata,flag)
      use alf, only: d => bnm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_alf_getdims_bnm
      subroutine f2py_alf_getdims_cm(r,s,f2pysetdata,flag)
      use alf, only: d => cm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_alf_getdims_cm
      subroutine f2py_alf_getdims_dnm(r,s,f2pysetdata,flag)
      use alf, only: d => dnm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_alf_getdims_dnm
      subroutine f2py_alf_getdims_anm(r,s,f2pysetdata,flag)
      use alf, only: d => anm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_alf_getdims_anm
      subroutine f2py_alf_getdims_narr(r,s,f2pysetdata,flag)
      use alf, only: d => narr

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_alf_getdims_narr
      subroutine f2py_alf_getdims_marr(r,s,f2pysetdata,flag)
      use alf, only: d => marr

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_alf_getdims_marr
      
      subroutine f2pyinitalf(f2pysetupfunc)
      use alf, only : en
      use alf, only : bnm
      use alf, only : cm
      use alf, only : dnm
      use alf, only : anm
      use alf, only : narr
      use alf, only : marr
      use alf, only : nmax0
      use alf, only : mmax0
      use alf, only : alfbasis
      use alf, only : initalf
      external f2pysetupfunc
      external f2py_alf_getdims_en
      external f2py_alf_getdims_bnm
      external f2py_alf_getdims_cm
      external f2py_alf_getdims_dnm
      external f2py_alf_getdims_anm
      external f2py_alf_getdims_narr
      external f2py_alf_getdims_marr
      call f2pysetupfunc(f2py_alf_getdims_en,f2py_alf_getdims_bnm,f2py_a&
     &lf_getdims_cm,f2py_alf_getdims_dnm,f2py_alf_getdims_anm,f2py_alf_g&
     &etdims_narr,f2py_alf_getdims_marr,nmax0,mmax0,alfbasis,initalf)
      end subroutine f2pyinitalf

      subroutine f2py_qwm_getdims_nb(r,s,f2pysetdata,flag)
      use qwm, only: d => nb

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_nb
      subroutine f2py_qwm_getdims_fs(r,s,f2pysetdata,flag)
      use qwm, only: d => fs

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_fs
      subroutine f2py_qwm_getdims_bm(r,s,f2pysetdata,flag)
      use qwm, only: d => bm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_bm
      subroutine f2py_qwm_getdims_mparm(r,s,f2pysetdata,flag)
      use qwm, only: d => mparm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_mparm
      subroutine f2py_qwm_getdims_fm(r,s,f2pysetdata,flag)
      use qwm, only: d => fm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_fm
      subroutine f2py_qwm_getdims_bz(r,s,f2pysetdata,flag)
      use qwm, only: d => bz

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_bz
      subroutine f2py_qwm_getdims_vnode(r,s,f2pysetdata,flag)
      use qwm, only: d => vnode

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_vnode
      subroutine f2py_qwm_getdims_zwght(r,s,f2pysetdata,flag)
      use qwm, only: d => zwght

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_zwght
      subroutine f2py_qwm_getdims_fl(r,s,f2pysetdata,flag)
      use qwm, only: d => fl

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_fl
      subroutine f2py_qwm_getdims_order(r,s,f2pysetdata,flag)
      use qwm, only: d => order

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_order
      subroutine f2py_qwm_getdims_tparm(r,s,f2pysetdata,flag)
      use qwm, only: d => tparm

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_qwm_getdims_tparm
      
      subroutine f2pyinitqwm(f2pysetupfunc)
      use qwm, only : lev
      use qwm, only : altsym
      use qwm, only : nbf
      use qwm, only : previous
      use qwm, only : nnode
      use qwm, only : maxs
      use qwm, only : nb
      use qwm, only : cseason
      use qwm, only : content
      use qwm, only : maxn
      use qwm, only : maxo
      use qwm, only : maxl
      use qwm, only : maxm
      use qwm, only : ctide
      use qwm, only : cwave
      use qwm, only : fs
      use qwm, only : altiso
      use qwm, only : tidefactor
      use qwm, only : bm
      use qwm, only : alttns
      use qwm, only : mparm
      use qwm, only : qwmdefault
      use qwm, only : e1
      use qwm, only : fm
      use qwm, only : bz
      use qwm, only : e2
      use qwm, only : wavefactor
      use qwm, only : h
      use qwm, only : vnode
      use qwm, only : p
      use qwm, only : priornb
      use qwm, only : component
      use qwm, only : zwght
      use qwm, only : nlev
      use qwm, only : qwminit
      use qwm, only : fl
      use qwm, only : order
      use qwm, only : tparm
      external f2pysetupfunc
      external f2py_qwm_getdims_nb
      external f2py_qwm_getdims_fs
      external f2py_qwm_getdims_bm
      external f2py_qwm_getdims_mparm
      external f2py_qwm_getdims_fm
      external f2py_qwm_getdims_bz
      external f2py_qwm_getdims_vnode
      external f2py_qwm_getdims_zwght
      external f2py_qwm_getdims_fl
      external f2py_qwm_getdims_order
      external f2py_qwm_getdims_tparm
      call f2pysetupfunc(lev,altsym,nbf,previous,nnode,maxs,f2py_qwm_get&
     &dims_nb,cseason,content,maxn,maxo,maxl,maxm,ctide,cwave,f2py_qwm_g&
     &etdims_fs,altiso,tidefactor,f2py_qwm_getdims_bm,alttns,f2py_qwm_ge&
     &tdims_mparm,qwmdefault,e1,f2py_qwm_getdims_fm,f2py_qwm_getdims_bz,&
     &e2,wavefactor,h,f2py_qwm_getdims_vnode,p,priornb,component,f2py_qw&
     &m_getdims_zwght,nlev,qwminit,f2py_qwm_getdims_fl,f2py_qwm_getdims_&
     &order,f2py_qwm_getdims_tparm)
      end subroutine f2pyinitqwm

      subroutine f2py_dwm_getdims_vshterms(r,s,f2pysetdata,flag)
      use dwm, only: d => vshterms

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_vshterms
      subroutine f2py_dwm_getdims_termarr(r,s,f2pysetdata,flag)
      use dwm, only: d => termarr

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_termarr
      subroutine f2py_dwm_getdims_dvbar(r,s,f2pysetdata,flag)
      use dwm, only: d => dvbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_dvbar
      subroutine f2py_dwm_getdims_mltterms(r,s,f2pysetdata,flag)
      use dwm, only: d => mltterms

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_mltterms
      subroutine f2py_dwm_getdims_dwbar(r,s,f2pysetdata,flag)
      use dwm, only: d => dwbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_dwbar
      subroutine f2py_dwm_getdims_coeff(r,s,f2pysetdata,flag)
      use dwm, only: d => coeff

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_coeff
      subroutine f2py_dwm_getdims_termval(r,s,f2pysetdata,flag)
      use dwm, only: d => termval

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_termval
      subroutine f2py_dwm_getdims_dpbar(r,s,f2pysetdata,flag)
      use dwm, only: d => dpbar

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_dwm_getdims_dpbar
      
      subroutine f2pyinitdwm(f2pysetupfunc)
      use dwm, only : nmax
      use dwm, only : vshterms
      use dwm, only : dtor
      use dwm, only : nvshterm
      use dwm, only : twidth
      use dwm, only : termarr
      use dwm, only : dvbar
      use dwm, only : dwmdefault
      use dwm, only : mltterms
      use dwm, only : dwbar
      use dwm, only : mmax
      use dwm, only : nterm
      use dwm, only : coeff
      use dwm, only : termval
      use dwm, only : dwminit
      use dwm, only : pi
      use dwm, only : dpbar
      external f2pysetupfunc
      external f2py_dwm_getdims_vshterms
      external f2py_dwm_getdims_termarr
      external f2py_dwm_getdims_dvbar
      external f2py_dwm_getdims_mltterms
      external f2py_dwm_getdims_dwbar
      external f2py_dwm_getdims_coeff
      external f2py_dwm_getdims_termval
      external f2py_dwm_getdims_dpbar
      call f2pysetupfunc(nmax,f2py_dwm_getdims_vshterms,dtor,nvshterm,tw&
     &idth,f2py_dwm_getdims_termarr,f2py_dwm_getdims_dvbar,dwmdefault,f2&
     &py_dwm_getdims_mltterms,f2py_dwm_getdims_dwbar,mmax,nterm,f2py_dwm&
     &_getdims_coeff,f2py_dwm_getdims_termval,dwminit,pi,f2py_dwm_getdim&
     &s_dpbar)
      end subroutine f2pyinitdwm

      subroutine f2py_gd2qdc_getdims_xcoeff(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => xcoeff

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_xcoeff
      subroutine f2py_gd2qdc_getdims_ycoeff(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => ycoeff

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_ycoeff
      subroutine f2py_gd2qdc_getdims_zcoeff(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => zcoeff

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_zcoeff
      subroutine f2py_gd2qdc_getdims_coeff(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => coeff

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1),s(2)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_coeff
      subroutine f2py_gd2qdc_getdims_shgradtheta(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => shgradtheta

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_shgradtheta
      subroutine f2py_gd2qdc_getdims_sh(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => sh

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_sh
      subroutine f2py_gd2qdc_getdims_normadj(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => normadj

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_normadj
      subroutine f2py_gd2qdc_getdims_shgradphi(r,s,f2pysetdata,flag)
      use gd2qdc, only: d => shgradphi

      integer flag
      external f2pysetdata
      logical ns
      integer r,i
      integer(8) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then
       allocate(d(s(1)))
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))
      end subroutine f2py_gd2qdc_getdims_shgradphi
      
      subroutine f2pyinitgd2qdc(f2pysetupfunc)
      use gd2qdc, only : nmax
      use gd2qdc, only : xcoeff
      use gd2qdc, only : dtor
      use gd2qdc, only : gd2qdinit
      use gd2qdc, only : ycoeff
      use gd2qdc, only : zcoeff
      use gd2qdc, only : alt
      use gd2qdc, only : mmax
      use gd2qdc, only : nterm
      use gd2qdc, only : coeff
      use gd2qdc, only : sineps
      use gd2qdc, only : shgradtheta
      use gd2qdc, only : epoch
      use gd2qdc, only : sh
      use gd2qdc, only : normadj
      use gd2qdc, only : pi
      use gd2qdc, only : shgradphi
      use gd2qdc, only : initgd2qd
      external f2pysetupfunc
      external f2py_gd2qdc_getdims_xcoeff
      external f2py_gd2qdc_getdims_ycoeff
      external f2py_gd2qdc_getdims_zcoeff
      external f2py_gd2qdc_getdims_coeff
      external f2py_gd2qdc_getdims_shgradtheta
      external f2py_gd2qdc_getdims_sh
      external f2py_gd2qdc_getdims_normadj
      external f2py_gd2qdc_getdims_shgradphi
      call f2pysetupfunc(nmax,f2py_gd2qdc_getdims_xcoeff,dtor,gd2qdinit,&
     &f2py_gd2qdc_getdims_ycoeff,f2py_gd2qdc_getdims_zcoeff,alt,mmax,nte&
     &rm,f2py_gd2qdc_getdims_coeff,sineps,f2py_gd2qdc_getdims_shgradthet&
     &a,epoch,f2py_gd2qdc_getdims_sh,f2py_gd2qdc_getdims_normadj,pi,f2py&
     &_gd2qdc_getdims_shgradphi,initgd2qd)
      end subroutine f2pyinitgd2qdc


