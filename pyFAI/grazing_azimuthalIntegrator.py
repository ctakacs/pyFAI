#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2019 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/07/2022"
__status__ = "stable"
__docformat__ = 'restructuredtext'

from pyFAI.grazing_geometry import GrazingGeometry

logger = logging.getLogger(__name__)



class GrazingAzimuthalIntegrator(GrazingGeometry):
    """

    """


    def __init__(self, dist=1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0,
             pixel1=0, pixel2=0, splinefile=None, detector=None,
             wavelength=None,
             useqx=True, sample_orientation=1, incident_angle=None,
             tilt_angle=0):

         GrazingGeometry.__init__(self, dist, poni1, poni2,
                           rot1, rot2, rot3,
                           pixel1, pixel2, splineFile, detector, wavelength, useqx, sample_orientation, incident_angle, tilt_angle)


        self._lock = threading.Semaphore()
        self.engines = {}  # key: name of the engine,

        self._empty = 0.0


    def reset(self):
        """
        Reset azimuthal integrator in addition to other arrays.
        """
        GrazingGeometry.reset(self)
        self.reset_engines()



















    def integrate2d_ng(self, data, npt_rad, npt_azim=360,
                        filename=None, correctSolidAngle=True, variance=None,
                        error_model=None, radial_range=None, azimuth_range=None,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method="bbox", unit=units.Q, safe=True,
                        normalization_factor=1.0, metadata=None):
        """
        Calculate the azimuthal regrouped 2d image in q(nm^-1)/chi(deg) by default

        Multi algorithm implementation (tries to be bullet proof)

        :param data: 2D array from the Detector/CCD camera
        :type data: ndarray
        :param npt_rad: number of points in the radial direction
        :type npt_rad: int
        :param npt_azim: number of points in the azimuthal direction
        :type npt_azim: int
        :param filename: output image (as edf format)
        :type filename: str
        :param correctSolidAngle: correct for solid angle of each pixel if True
        :type correctSolidAngle: bool
        :param variance: array containing the variance of the data. If not available, no error propagation is done
        :type variance: ndarray
        :param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :type error_model: str
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional
        :param mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor between -1 (vertical)
                and +1 (horizontal). 0 for circular polarization or random,
                None for no correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray
        :param method: can be "numpy", "cython", "BBox" or "splitpixel", "lut", "csr; "lut_ocl" and "csr_ocl" if you want to go on GPU. To Specify the device: "csr_ocl_1,2"
        :type method: str
        :param unit: Output units, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for now
        :type unit: pyFAI.units.Unit
        :param safe: Do some extra checks to ensure LUT is still valid. False is faster.
        :type safe: bool
        :param normalization_factor: Value of a normalization monitor
        :type normalization_factor: float
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :return: azimuthaly regrouped intensity, q/2theta/r pos. and chi pos.
        :rtype: Integrate2dResult, dict
        """
        method = self._normalize_method(method, dim=2, default=self.DEFAULT_METHOD_2D)
        assert method.dimension == 2
        npt = (npt_rad, npt_azim)
        unit = units.to_unit(unit)
        pos0_scale = unit.scale
        empty = dummy if dummy is not None else self._empty
        if mask is None:
            has_mask = "from detector"
            mask = self.mask
            mask_crc = self.detector.get_mask_crc()
            if mask is None:
                has_mask = False
                mask_crc = None
        else:
            has_mask = "provided"
            mask = numpy.ascontiguousarray(mask)
            mask_crc = crc32(mask)

        shape = data.shape

        if radial_range:
            radial_range = tuple([i / pos0_scale for i in radial_range])

        if variance is not None:
            assert variance.size == data.size
        elif error_model:
            error_model = error_model.lower()
            if error_model == "poisson":
                variance = numpy.ascontiguousarray(data, numpy.float32)

        if azimuth_range is not None:
            azimuth_range = tuple(deg2rad(azimuth_range[i]) for i in (0, -1))
            if azimuth_range[1] <= azimuth_range[0]:
                azimuth_range = (azimuth_range[0], azimuth_range[1] + 2 * pi)
            self.check_chi_disc(azimuth_range)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = polarization_crc = None
        else:
            polarization, polarization_crc = self.polarization(shape, polarization_factor, with_checksum=True)

        if dark is None:
            dark = self.detector.darkcurrent
            if dark is None:
                has_dark = False
            else:
                has_dark = "from detector"
        else:
            has_dark = "provided"

        if flat is None:
            flat = self.detector.flatfield
            if dark is None:
                has_flat = False
            else:
                has_flat = "from detector"
        else:
            has_flat = "provided"

        I = None
        sigma = None
        sum_ = None
        count = None
        signal2d = None
        norm2d = None
        var2d = None

        if method.algo_lower in ("csr", "lut"):
            intpl = None
            cython_method = IntegrationMethod.select_method(method.dimension, method.split_lower, method.algo_lower, "cython")[0]
            if cython_method not in self.engines:
                cython_engine = self.engines[cython_method] = Engine()
            else:
                cython_engine = self.engines[cython_method]
            with cython_engine.lock:
                cython_integr = cython_engine.engine
                cython_reset = None

                if cython_integr is None:
                    cython_reset = "of first initialization"
                if (not cython_reset) and safe:
                    if cython_integr.unit != unit:
                        cython_reset = "unit was changed"
                    if cython_integr.bins != npt:
                        cython_reset = "number of points changed"
                    if cython_integr.size != data.size:
                        cython_reset = "input image size changed"
                    if cython_integr.empty != empty:
                        cython_reset = "empty value changed"
                    if (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = f"mask but {method.algo_lower.upper()} was without mask"
                    elif (mask is None) and (cython_integr.cmask is not None):
                        cython_reset = f"no mask but { method.algo_lower.upper()} has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    if (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = f"radial_range was defined in { method.algo_lower.upper()}"
                    elif (radial_range is not None) and (cython_integr.pos0_range != radial_range):
                        cython_reset = f"radial_range is defined but differs in %s" % method.algo_lower.upper()
                    if (azimuth_range is None) and (cython_integr.pos1_range is not None):
                        cython_reset = f"azimuth_range not defined and {method.algo_lower.upper()} had azimuth_range defined"
                    elif (azimuth_range is not None) and (cython_integr.pos1_range != azimuth_range):
                        cython_reset = f"azimuth_range requested and {method.algo_lower.upper()}'s azimuth_range don't match"
                if cython_reset:
                    logger.info("AI.integrate2d_ng: Resetting Cython integrator because %s", cython_reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        if method.algo_lower == "csr":
                            cython_integr = self.setup_CSR(shape, npt, mask,
                                                           radial_range, azimuth_range,
                                                           mask_checksum=mask_crc,
                                                           unit=unit, split=split,
                                                           empty=empty, scale=False)
                        else:
                            cython_integr = self.setup_LUT(shape, npt, mask,
                                                           radial_range, azimuth_range,
                                                           mask_checksum=mask_crc,
                                                           unit=unit, split=split,
                                                           empty=empty, scale=False)
                    except MemoryError:  # CSR method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        cython_integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_1D
                    else:
                        cython_engine.set_engine(cython_integr)
            # This whole block uses CSR, Now we should treat all the various implementation: Cython, OpenCL and finally Python.
            if method.impl_lower != "cython":
                # method.impl_lower in ("opencl", "python"):
                if method not in self.engines:
                    # instanciated the engine
                    engine = self.engines[method] = Engine()
                else:
                    engine = self.engines[method]
                with engine.lock:
                    # Validate that the engine used is the proper one
                    integr = engine.engine
                    reset = None
                    if integr is None:
                        reset = "init"
                    if (not reset) and safe:
                        if integr.unit != unit:
                            reset = "unit changed"
                        if integr.bins != numpy.prod(npt):
                            reset = "number of points changed"
                        if integr.size != data.size:
                            reset = "input image size changed"
                        if integr.empty != empty:
                            reset = "empty value changed"
                        if (mask is not None) and (not integr.check_mask):
                            reset = "mask but CSR was without mask"
                        elif (mask is None) and (integr.check_mask):
                            reset = "no mask but CSR has mask"
                        elif (mask is not None) and (integr.mask_checksum != mask_crc):
                            reset = "mask changed"
                        if (radial_range is None) and (integr.pos0_range is not None):
                            reset = "radial_range was defined in CSR"
                        elif (radial_range is not None) and integr.pos0_range != (min(radial_range), max(radial_range)):
                            reset = "radial_range is defined but differs in CSR"
                        if (azimuth_range is None) and (integr.pos1_range is not None):
                            reset = "azimuth_range not defined and CSR had azimuth_range defined"
                        elif (azimuth_range is not None) and integr.pos1_range != (min(azimuth_range), max(azimuth_range)):
                            reset = "azimuth_range requested and CSR's azimuth_range don't match"
                    error = False
                    if reset:
                        logger.info("AI.integrate2d: Resetting integrator because %s", reset)
                        split = method.split_lower
                        try:
                            if method.algo_lower == "csr":
                                cython_integr = self.setup_CSR(shape, npt, mask,
                                                               radial_range, azimuth_range,
                                                               mask_checksum=mask_crc,
                                                               unit=unit, split=split,
                                                               empty=empty, scale=False)
                            else:
                                cython_integr = self.setup_LUT(shape, npt, mask,
                                                               radial_range, azimuth_range,
                                                               mask_checksum=mask_crc,
                                                               unit=unit, split=split,
                                                               empty=empty, scale=False)
                        except MemoryError:
                            logger.warning("MemoryError: falling back on default implementation")
                            cython_integr = None
                            self.reset_engines()
                            method = self.DEFAULT_METHOD_2D
                            error = True
                        else:
                            error = False
                            cython_engine.set_engine(cython_integr)
                if not error:
                    if method in self.engines:
                        ocl_py_engine = self.engines[method]
                    else:
                        ocl_py_engine = self.engines[method] = Engine()
                    integr = ocl_py_engine.engine
                    if integr is None or integr.checksum != cython_integr.lut_checksum:
                        if (method.impl_lower == "opencl"):
                            with ocl_py_engine.lock:
                                integr = method.class_funct_ng.klass(cython_integr.lut,
                                                                     cython_integr.size,
                                                                     bin_centers=cython_integr.bin_centers0,
                                                                     azim_centers=cython_integr.bin_centers1,
                                                                     platformid=method.target[0],
                                                                     deviceid=method.target[1],
                                                                     checksum=cython_integr.lut_checksum,
                                                                     unit=unit, empty=empty,
                                                                     mask_checksum=mask_crc
                                                                     )

                        elif (method.impl_lower == "python"):
                            with ocl_py_engine.lock:
                                integr = method.class_funct_ng.klass(cython_integr.lut,
                                                                     cython_integr.size,
                                                                     bin_centers=cython_integr.bin_centers0,
                                                                     azim_centers=cython_integr.bin_centers1,
                                                                     checksum=cython_integr.lut_checksum,
                                                                     unit=unit, empty=empty,
                                                                     mask_checksum=mask_crc)
                        ocl_py_engine.set_engine(integr)

                    if (integr is not None):
                            intpl = integr.integrate_ng(data,
                                                       variance=variance,
                                                       dark=dark, flat=flat,
                                                       solidangle=solidangle,
                                                       solidangle_checksum=self._dssa_crc,
                                                       dummy=dummy,
                                                       delta_dummy=delta_dummy,
                                                       polarization=polarization,
                                                       polarization_checksum=polarization_crc,
                                                       safe=safe,
                                                       normalization_factor=normalization_factor)
            if intpl is None:  # fallback if OpenCL failed or default cython
                # The integrator has already been initialized previously
                intpl = cython_integr.integrate_ng(data,
                                                   variance=variance,
                                                   # poissonian=poissonian,
                                                   dummy=dummy,
                                                   delta_dummy=delta_dummy,
                                                   dark=dark,
                                                   flat=flat,
                                                   solidangle=solidangle,
                                                   polarization=polarization,
                                                   normalization_factor=normalization_factor)
            I = intpl.intensity
            bins_rad = intpl.radial
            bins_azim = intpl.azimuthal
            signal2d = intpl.signal
            norm2d = intpl.normalization
            count = intpl.count
            if variance is not None:
                sigma = intpl.sigma
                var2d = intpl.variance

        elif method.algo_lower == "histogram":
            if method.split_lower in ("pseudo", "full"):
                logger.debug("integrate2d uses (full, histogram, cython) implementation")
                pos = self.array_from_unit(shape, "corner", unit, scale=False)
                integrator = method.class_funct_ng.function
                intpl = integrator(pos=pos,
                                 weights=data,
                                 bins=(npt_rad, npt_azim),
                                 pos0_range=radial_range,
                                 pos1_range=azimuth_range,
                                 dummy=dummy,
                                 delta_dummy=delta_dummy,
                                 mask=mask,
                                 dark=dark,
                                 flat=flat,
                                 solidangle=solidangle,
                                 polarization=polarization,
                                 normalization_factor=normalization_factor,
                                 chiDiscAtPi=self.chiDiscAtPi,
                                 empty=empty,
                                 variance=variance)

            elif method.split_lower == "bbox":
                logger.debug("integrate2d uses BBox implementation")
                chi = self.chiArray(shape)
                dchi = self.deltaChi(shape)
                pos0 = self.array_from_unit(shape, "center", unit, scale=False)
                dpos0 = self.array_from_unit(shape, "delta", unit, scale=False)
                intpl = splitBBox.histoBBox2d_ng(weights=data,
                                               pos0=pos0,
                                               delta_pos0=dpos0,
                                               pos1=chi,
                                               delta_pos1=dchi,
                                               bins=(npt_rad, npt_azim),
                                               pos0_range=radial_range,
                                               pos1_range=azimuth_range,
                                               dummy=dummy,
                                               delta_dummy=delta_dummy,
                                               mask=mask,
                                               dark=dark,
                                               flat=flat,
                                               solidangle=solidangle,
                                               polarization=polarization,
                                               normalization_factor=normalization_factor,
                                               chiDiscAtPi=self.chiDiscAtPi,
                                               empty=empty,
                                               variance=variance)
            elif method.split_lower == "no":
                if method.impl_lower == "opencl":
                    logger.debug("integrate2d uses OpenCL histogram implementation")
                    if method not in self.engines:
                    # instanciated the engine
                        engine = self.engines[method] = Engine()
                    else:
                        engine = self.engines[method]
                    with engine.lock:
                        # Validate that the engine used is the proper one #TODO!!!!
                        integr = engine.engine
                        reset = None
                        if integr is None:
                            reset = "init"
                        if (not reset) and safe:
                            if integr.unit != unit:
                                reset = "unit changed"
                            if (integr.bins_radial, integr.bins_azimuthal) != npt:
                                reset = "number of points changed"
                            if integr.size != data.size:
                                reset = "input image size changed"
                            if (mask is not None) and (not integr.check_mask):
                                reset = "mask but CSR was without mask"
                            elif (mask is None) and (integr.check_mask):
                                reset = "no mask but CSR has mask"
                            elif (mask is not None) and (integr.on_device.get("mask") != mask_crc):
                                reset = "mask changed"
                            if self._cached_array[unit.name.split("_")[0] + "_crc"] != integr.on_device.get("radial"):
                                reset = "radial array changed"
                            if self._cached_array["chi_crc"] != integr.on_device.get("azimuthal"):
                                reset = "azimuthal array changed"
                            # Nota: Ranges are enforced at runtime, not initialization
                        error = False
                        if reset:
                            logger.info("AI.integrate2d: Resetting OCL_Histogram2d integrator because %s", reset)
                            rad = self.array_from_unit(shape, typ="center", unit=unit, scale=False)
                            rad_crc = self._cached_array[unit.name.split("_")[0] + "_crc"] = crc32(rad)
                            azi = self.chiArray(shape)
                            azi_crc = self._cached_array["chi_crc"] = crc32(azi)
                            try:
                                integr = method.class_funct_ng.klass(rad,
                                                                     azi,
                                                                     *npt,
                                                                     radial_checksum=rad_crc,
                                                                     azimuthal_checksum=azi_crc,
                                                                     empty=empty, unit=unit,
                                                                     mask=mask, mask_checksum=mask_crc,
                                                                     platformid=method.target[0],
                                                                     deviceid=method.target[1]
                                                                     )
                            except MemoryError:
                                logger.warning("MemoryError: falling back on default forward implementation")
                                integr = None
                                self.reset_engines()
                                method = self.DEFAULT_METHOD_2D
                                error = True
                            else:
                                error = False
                                engine.set_engine(integr)
                    if not error:
                        intpl = integr.integrate(data, dark=dark, flat=flat,
                                                 solidangle=solidangle,
                                                 solidangle_checksum=self._dssa_crc,
                                                 dummy=dummy,
                                                 delta_dummy=delta_dummy,
                                                 polarization=polarization,
                                                 polarization_checksum=polarization_crc,
                                                 safe=safe,
                                                 normalization_factor=normalization_factor,
                                                 radial_range=radial_range,
                                                 azimuthal_range=azimuth_range)
###################3
                elif method.impl_lower == "cython":
                    logger.debug("integrate2d uses Cython histogram implementation")
                    prep = preproc(data,
                                   dark=dark,
                                   flat=flat,
                                   solidangle=solidangle,
                                   polarization=polarization,
                                   absorption=None,
                                   mask=mask,
                                   dummy=dummy,
                                   delta_dummy=delta_dummy,
                                   normalization_factor=normalization_factor,
                                   empty=self._empty,
                                   split_result=4,
                                   variance=variance,
                                   # dark_variance=None,
                                   # poissonian=False,
                                   dtype=numpy.float32)
                    pos0 = self.array_from_unit(shape, "center", unit, scale=False)
                    chi = self.chiArray(shape)
                    intpl = histogram.histogram2d_engine(pos0=pos0,
                                                       pos1=chi,
                                                       weights=prep,
                                                       bins=(npt_rad, npt_azim),
                                                       pos0_range=radial_range,
                                                       pos1_range=azimuth_range,
                                                       split=False,
                                                       empty=empty,
                                                       )

                else:  # Python implementation:
                    logger.debug("integrate2d uses python implementation")
                    data = data.astype(numpy.float32)  # it is important to make a copy see issue #88
                    mask = self.create_mask(data, mask, dummy, delta_dummy,
                                            unit=unit,
                                            radial_range=radial_range,
                                            azimuth_range=azimuth_range,
                                            mode="normal").ravel()
                    pos0 = self.array_from_unit(shape, "center", unit, scale=False).ravel()
                    pos1 = self.chiArray(shape).ravel()

                    if radial_range is None:
                        radial_range = [pos0.min(), pos0.max()]
                    if azimuth_range is None:
                        azimuth_range = [pos1.min(), pos1.max()]

                    if method.method[1:4] == ("no", "histogram", "python"):
                        logger.debug("integrate2d uses Numpy implementation")
                        intpl = histogram_engine.histogram2d_engine(radial=pos0,
                                                                    azimuthal=pos1,
                                                                    npt=(npt_rad, npt_azim),
                                                                    raw=data,
                                                                    dark=dark,
                                                                    flat=flat,
                                                                    solidangle=solidangle,
                                                                    polarization=polarization,
                                                                    absorption=None,
                                                                    mask=mask,
                                                                    dummy=dummy,
                                                                    delta_dummy=delta_dummy,
                                                                    normalization_factor=normalization_factor,
                                                                    empty=self._empty,
                                                                    split_result=False,
                                                                    variance=variance,
                                                                    dark_variance=None,
                                                                    error_model=ErrorModel.NO,
                                                                    radial_range=radial_range,
                                                                    azimuth_range=azimuth_range)
            I = intpl.intensity
            bins_azim = intpl.azimuthal
            bins_rad = intpl.radial
            signal2d = intpl.signal
            norm2d = intpl.normalization
            count = intpl.count
            if variance is not None:
                sigma = intpl.sigma
                var2d = intpl.variance

        # Duplicate arrays on purpose ....
        bins_rad = bins_rad * pos0_scale
        bins_azim = bins_azim * (180.0 / pi)

        result = Integrate2dResult(I, bins_rad, bins_azim, sigma)
        result._set_method_called("integrate2d")
        result._set_compute_engine(str(method))
        result._set_method(method)
        result._set_unit(unit)
        result._set_count(count)
        result._set_sum(sum_)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_has_mask_applied(has_mask)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_metadata(metadata)

        result._set_sum_signal(signal2d)
        result._set_sum_normalization(norm2d)
        result._set_sum_variance(var2d)

        if filename is not None:
            writer = DefaultAiWriter(filename, self)
            writer.write(result)

        return result

    integrate2d = _integrate2d_ng = integrate2d_ng
