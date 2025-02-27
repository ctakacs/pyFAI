#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"Test suite for worker"

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/04/2022"

import unittest
import logging
import os.path
import shutil
import numpy

from silx.io.url import DataUrl

from .. import units
from .. import worker as worker_mdl
from ..worker import Worker, PixelwiseWorker
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..containers import Integrate1dResult
from ..containers import Integrate2dResult
from . import utilstest

logger = logging.getLogger(__name__)


class AzimuthalIntegratorMocked():

    def __init__(self, result=None):
        self._integrate1d_called = 0
        self._integrate2d_called = 0
        self._result = result
        self._csr_integrator = 0
        self._lut_integrator = numpy.array([0])

    def integrate1d(self, **kargs):
        self._integrate1d_called += 1
        self._integrate1d_kargs = kargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    integrate1d_ng = integrate1d_legacy = integrate1d

    def integrate2d(self, **kargs):
        self._integrate2d_called += 1
        self._integrate2d_kargs = kargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    integrate2d_ng = integrate2d_legacy = integrate2d


class MockedAiWriter():

    def __init__(self, result=None):
        self._write_called = 0
        self._close_called = 0
        self._write_kargs = {}

    def write(self, data):
        self._write_called += 1
        self._write_kargs["data"] = data

    def close(self):
        self._close_called += 1


class TestWorker(unittest.TestCase):

    def test_constructor_ai(self):
        ai = AzimuthalIntegrator()
        w = Worker(ai)
        self.assertIsNotNone(w)

    def test_constructor(self):
        w = Worker()
        self.assertIsNotNone(w)

    def test_process_1d(self):
        ai_result = Integrate1dResult(numpy.array([0, 1]), numpy.array([2, 3]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(1, 10))
        data = numpy.array([0])
        worker.unit = units.TTH_DEG
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "lut"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        worker.output = "numpy"
        worker.update_processor()
        result = worker.process(data)

        # ai calls
        self.assertEqual(ai._integrate1d_called, 1)
        self.assertEqual(ai._integrate2d_called, 0)
        ai_args = ai._integrate1d_kargs
        self.assertEqual(ai_args["unit"], worker.unit)
        self.assertEqual(ai_args["dummy"], worker.dummy)
        self.assertEqual(ai_args["delta_dummy"], worker.delta_dummy)
        self.assertTrue(worker.method in str(ai_args["method"]).lower())
        self.assertEqual(ai_args["polarization_factor"], worker.polarization_factor)
        self.assertEqual(ai_args["safe"], True)
        self.assertEqual(ai_args["data"], data)
        self.assertEqual(ai_args["correctSolidAngle"], worker.correct_solid_angle)
        self.assertEqual(ai_args["npt"], worker.nbpt_rad)

        # result
        self.assertEqual(result.tolist(), [[0, 2], [1, 3]])
        self.assertEqual(worker.radial.tolist(), [0, 1])
        self.assertEqual(worker.azimuthal, None)

    def test_process_2d(self):
        ai_result = Integrate2dResult(numpy.array([0]), numpy.array([1]), numpy.array([2]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(10, 10))
        data = numpy.array([0])
        worker.unit = units.TTH_RAD
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "lut"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        worker.output = "numpy"
        worker.update_processor()
        result = worker.process(data)

        # ai calls
        self.assertEqual(ai._integrate1d_called, 0)
        self.assertEqual(ai._integrate2d_called, 1)
        ai_args = ai._integrate2d_kargs
        self.assertEqual(ai_args["unit"], worker.unit)
        self.assertEqual(ai_args["dummy"], worker.dummy)
        self.assertEqual(ai_args["delta_dummy"], worker.delta_dummy)
        self.assertTrue(worker.method in str(ai_args["method"]).lower())
        self.assertEqual(ai_args["polarization_factor"], worker.polarization_factor)
        self.assertEqual(ai_args["safe"], True)
        self.assertEqual(ai_args["data"], data)
        self.assertEqual(ai_args["correctSolidAngle"], worker.correct_solid_angle)
        self.assertEqual(ai_args["npt_rad"], worker.nbpt_rad)
        self.assertEqual(ai_args["npt_azim"], worker.nbpt_azim)

        # result
        self.assertEqual(result.tolist(), [0])
        self.assertEqual(worker.radial.tolist(), [1])
        self.assertEqual(worker.azimuthal.tolist(), [2])

    def test_1d_writer(self):
        ai_result = Integrate1dResult(numpy.array([0]), numpy.array([1]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        data = numpy.array([0])
        worker = Worker(ai, shapeOut=(1, 10))
        worker.unit = units.TTH_DEG
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "d"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        # worker.nbpt_azim = 1
        worker.output = "numpy"

        writer = MockedAiWriter()
        _result = worker.process(data, writer=writer)

        self.assertEqual(writer._write_called, 1)
        self.assertEqual(writer._close_called, 0)
        self.assertIs(writer._write_kargs["data"], ai_result)

    def test_2d_writer(self):
        ai_result = Integrate2dResult(numpy.array([0]), numpy.array([1]), numpy.array([2]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        data = numpy.array([0])
        worker = Worker(ai, shapeOut=(1, 10))
        worker.unit = units.TTH_DEG
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "d"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        # worker.nbpt_azim = 1
        worker.output = "numpy"

        writer = MockedAiWriter()
        _result = worker.process(data, writer=writer)

        self.assertEqual(writer._write_called, 1)
        self.assertEqual(writer._close_called, 0)
        self.assertIs(writer._write_kargs["data"], ai_result)

    def test_process_exception(self):
        ai = AzimuthalIntegratorMocked(result=Exception("Out of memory"))
        worker = Worker(ai)
        data = numpy.array([0])
        # worker.nbpt_azim = 2
        try:
            worker.process(data)
        except Exception:
            pass

    def test_process_error_model(self):
        ai_result = Integrate1dResult(numpy.array([0]), numpy.array([1]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(1, 10))
        data = numpy.array([0])
        # worker.nbpt_azim = 1
        for error_model in ["poisson", "azimuthal", None]:
            worker.error_model = error_model
            worker.process(data)
            if error_model:
                self.assertIn("error_model", ai._integrate1d_kargs)
                self.assertEqual(ai._integrate1d_kargs["error_model"], error_model)
            else:
                self.assertNotIn("error_model", ai._integrate1d_kargs)

    def test_process_no_output(self):
        ai_result = Integrate1dResult(numpy.array([0]), numpy.array([1]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(1, 10))
        data = numpy.array([0])
        worker.output = None
        # worker.nbpt_azim = 1
        worker.do_poisson = True
        result = worker.process(data)
        self.assertIsNone(result)

    def test_pixelwiseworker(self):
        shape = (5, 7)
        size = numpy.prod(shape)
        ref = numpy.random.randint(0, 60000, size=size).reshape(shape).astype("uint16")
        dark = numpy.random.poisson(10, size=size).reshape(shape).astype("uint16")
        raw = ref + dark
        flat = numpy.random.normal(1.0, 0.1, size=size).reshape(shape)
        signal = ref / flat
        precision = 1e-2

        # Without error propagation
        # Numpy path
        worker_mdl.USE_CYTHON = False
        pww = PixelwiseWorker(dark=dark, flat=flat, dummy=-5, dtype="float64")
        res_np = pww.process(raw, normalization_factor=6.0)
        err = abs(res_np - signal / 6.0).max()
        self.assertLess(err, precision, "Numpy calculation are OK: %s" % err)

        # Cython path
        worker_mdl.USE_CYTHON = True
        res_cy = pww.process(raw, normalization_factor=7.0)
        err = abs(res_cy - signal / 7.0).max()
        self.assertLess(err, precision, "Cython calculation are OK: %s" % err)

        # With Poissonian errors
        # Numpy path
        worker_mdl.USE_CYTHON = False
        pww = PixelwiseWorker(dark=dark)
        res_np, err_np = pww.process(raw, variance=ref, normalization_factor=2.0)
        delta_res = abs(res_np - ref / 2.0).max()
        delta_err = abs(err_np - numpy.sqrt(ref) / 2.0).max()

        self.assertLess(delta_res, precision, "Numpy intensity calculation are OK: %s" % err)
        self.assertLess(delta_err, precision, "Numpy error calculation are OK: %s" % err)

        # Cython path
        worker_mdl.USE_CYTHON = True
        res_cy, err_cy = pww.process(raw, variance=ref, normalization_factor=2.0)
        delta_res = abs(res_cy - ref / 2.0).max()
        delta_err = abs(err_cy - numpy.sqrt(ref) / 2.0).max()
        self.assertLess(delta_res, precision, "Cython intensity calculation are OK: %s" % err)
        self.assertLess(delta_err, precision, "Cython error calculation are OK: %s" % err)


class TestWorkerConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = os.path.join(utilstest.test_options.tempdir, cls.__name__)
        os.makedirs(cls.directory)

        cls.a = os.path.join(cls.directory, "a.npy")
        cls.b = os.path.join(cls.directory, "b.npy")
        cls.c = os.path.join(cls.directory, "c.npy")
        cls.d = os.path.join(cls.directory, "d.npy")

        cls.shape = (2, 2)
        ones = numpy.ones(shape=cls.shape)
        numpy.save(cls.a, ones)
        numpy.save(cls.b, ones * 2)
        numpy.save(cls.c, ones * 3)
        numpy.save(cls.d, ones * 4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory)

    def test_flatdark(self):
        config = {"version": 2,
                  "application": "pyfai-integrate",
                  "dark_current": [self.a, self.b, self.c],
                  "flat_field": [self.a, self.b, self.d],
                  "poni": utilstest.UtilsTest.getimage("Pilatus1M.poni"),
                  "detector": "Detector",
                  "detector_config": {"pixel1": 1, "pixel2": 1, "max_shape": (2, 2)},
                  "do_2D": False,
                  "nbpt_rad": 2,
                  "do_solid_angle": False,
                  "method": "splitbbox"}
        worker = Worker()
        worker.validate_config(config)
        worker.set_config(config)
        data = numpy.ones(shape=self.shape)
        worker.process(data=data)
        self.assertTrue(numpy.isclose(worker.ai.detector.get_darkcurrent()[0, 0], (1 + 2 + 3) / 3))
        self.assertTrue(numpy.isclose(worker.ai.detector.get_flatfield()[0, 0], (1 + 2 + 4) / 3))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestWorker))
    testsuite.addTest(loader(TestWorkerConfig))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
