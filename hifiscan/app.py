import asyncio
import datetime as dt
import logging
import os
import signal
import sys
from pathlib import Path

import numpy as np
from PySide6 import QtGui as qtgui, QtWidgets as qt
import pyqtgraph as pg

import hifiscan as hifi


class App(qt.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('HiFi Scan')
        topWidget = qt.QWidget()
        self.setCentralWidget(topWidget)
        vbox = qt.QVBoxLayout()
        topWidget.setLayout(vbox)

        self.stack = qt.QStackedWidget()
        self.stack.addWidget(self.createSpectrumWidget())
        self.stack.addWidget(self.createIRWidget())
        self.stack.currentChanged.connect(self.plot)
        vbox.addWidget(self.stack)
        vbox.addWidget(self.createSharedControls())

        self.paused = False
        self.analyzer = None
        self.refAnalyzer = None
        self.saveDir = Path.home()
        self.loop = asyncio.get_event_loop_policy().get_event_loop()
        self.task = self.loop.create_task(wrap_coro(self.analyze()))

        self.resize(1800, 900)
        self.show()

    async def analyze(self):
        with hifi.Audio() as audio:
            while True:
                lo = self.lo.value()
                hi = self.hi.value()
                secs = self.secs.value()
                ampl = self.ampl.value() / 100
                if self.paused or lo >= hi or secs <= 0 or not ampl:
                    await asyncio.sleep(0.1)
                    continue

                analyzer = hifi.Analyzer(lo, hi, secs, audio.rate, ampl)
                audio.play(analyzer.chirp)
                async for recording in audio.record():
                    if self.paused:
                        audio.cancelPlay()
                        break
                    if analyzer.findMatch(recording):
                        self.analyzer = analyzer
                        self.plot()
                        break
                    if analyzer.timedOut():
                        break

    def setPaused(self):
        self.paused = not self.paused

    def plot(self, *_):
        if self.stack.currentIndex() == 0:
            self.plotSpectrum()
        else:
            self.plotIR()

    def plotSpectrum(self):
        smoothing = self.spectrumSmoothing.value()
        if self.analyzer:
            spectrum = self.analyzer.spectrum(smoothing)
            self.spectrumPlot.setData(*spectrum)
        if self.refAnalyzer:
            spectrum = self.refAnalyzer.spectrum(smoothing)
            self.refSpectrumPlot.setData(*spectrum)

    def plotIR(self):
        if self.refAnalyzer and self.useRefBox.isChecked():
            analyzer = self.refAnalyzer
        else:
            analyzer = self.analyzer
        if not analyzer:
            return
        secs = self.msDuration.value() / 1000
        dbRange = self.dbRange.value()
        beta = self.kaiserBeta.value()
        smoothing = self.irSmoothing.value()

        t, ir = analyzer.h_inv(secs, dbRange, beta, smoothing)
        self.irPlot.setData(1000 * t, ir)

        logIr = np.log10(1e-8 + np.abs(ir))
        self.logIrPlot.setData(1000 * t, logIr)

        corrFactor = analyzer.correctionFactor(ir)
        self.correctionPlot.setData(*corrFactor)

        spectrum, spectrum_resamp = analyzer.correctedSpectrum(corrFactor)
        self.simPlot.setData(*spectrum)
        self.avSimPlot.setData(*spectrum_resamp)

    def screenshot(self):
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'hifiscan_{timestamp}.png'
        filename, _ = qt.QFileDialog.getSaveFileName(
            self, 'Save screenshot', str(self.saveDir / name), 'PNG (*.png)')
        if filename:
            self.stack.grab().save(filename)
            self.saveDir = Path(filename).parent

    def saveIR(self):
        if self.refAnalyzer and self.useRefBox.isChecked():
            analyzer = self.refAnalyzer
        else:
            analyzer = self.analyzer
        if not analyzer:
            return
        ms = int(self.msDuration.value())
        db = int(self.dbRange.value())
        beta = int(self.kaiserBeta.value())
        smoothing = int(self.irSmoothing.value())
        _, irInv = analyzer.h_inv(ms / 1000, db, beta, smoothing)

        name = f'IR_{ms}ms_{db}dB_{beta}t_{smoothing}s.wav'
        filename, _ = qt.QFileDialog.getSaveFileName(
            self, 'Save inverse impulse response',
            str(self.saveDir / name), 'WAV (*.wav)')
        if filename:
            hifi.write_wav(filename, analyzer.rate, irInv)
            self.saveDir = Path(filename).parent

    def setReference(self, withRef: bool):
        if withRef:
            if self.analyzer:
                self.refAnalyzer = self.analyzer
                self.plot()
        else:
            self.refAnalyzer = None
            self.refSpectrumPlot.clear()
            self.spectrumPlotWidget.repaint()

    def run(self):
        """Run both the Qt and asyncio event loops."""

        def updateQt():
            qApp = qtgui.QGuiApplication.instance()
            qApp.processEvents()
            self.loop.call_later(0.03, updateQt)

        signal.signal(signal.SIGINT, lambda *args: self.close())
        updateQt()
        self.loop.run_forever()
        self.loop.run_until_complete(self.task)
        os._exit(0)

    def closeEvent(self, ev):
        self.task.cancel()
        self.loop.stop()

    def createSpectrumWidget(self) -> qt.QWidget:
        topWidget = qt.QWidget()
        vbox = qt.QVBoxLayout()
        topWidget.setLayout(vbox)

        axes = {ori: Axis(ori) for ori in
                ['bottom', 'left', 'top', 'right']}
        for ax in axes.values():
            ax.setGrid(255)
        self.spectrumPlotWidget = pw = pg.PlotWidget(axisItems=axes)
        pw.setLabel('left', 'Relative Power [dB]')
        pw.setLabel('bottom', 'Frequency [Hz]')
        pw.setLogMode(x=True)
        self.refSpectrumPlot = pw.plot(pen=(255, 100, 0), stepMode='right')
        self.spectrumPlot = pw.plot(pen=(0, 255, 255), stepMode='right')
        self.spectrumPlot.curve.setCompositionMode(
            qtgui.QPainter.CompositionMode_Plus)
        vbox.addWidget(pw)

        self.lo = pg.SpinBox(
            value=20, step=5, bounds=[5, 40000], suffix='Hz')
        self.hi = pg.SpinBox(
            value=20000, step=100, bounds=[5, 40000], suffix='Hz')
        self.secs = pg.SpinBox(
            value=1.0, step=0.1, bounds=[0.1, 10], suffix='s')
        self.ampl = pg.SpinBox(
            value=40, step=1, bounds=[0, 100], suffix='%')
        self.spectrumSmoothing = pg.SpinBox(
            value=15, step=1, bounds=[0, 30])
        self.spectrumSmoothing.sigValueChanging.connect(self.plot)
        refBox = qt.QCheckBox('Reference')
        refBox.stateChanged.connect(self.setReference)

        hbox = qt.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(qt.QLabel('Low: '))
        hbox.addWidget(self.lo)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('High: '))
        hbox.addWidget(self.hi)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Duration: '))
        hbox.addWidget(self.secs)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Amplitude: '))
        hbox.addWidget(self.ampl)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Smoothing: '))
        hbox.addWidget(self.spectrumSmoothing)
        hbox.addSpacing(32)
        hbox.addWidget(refBox)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        return topWidget

    def createIRWidget(self) -> qt.QWidget:
        topWidget = qt.QWidget()
        vbox = qt.QVBoxLayout()
        topWidget.setLayout(vbox)
        splitter = qt.QSplitter(qtgui.Qt.Vertical)
        vbox.addWidget(splitter)

        self.irPlotWidget = pw = pg.PlotWidget()
        pw.showGrid(True, True, 0.8)
        self.irPlot = pw.plot(pen=(0, 255, 255))
        pw.setLabel('left', 'Inverse IR')
        splitter.addWidget(pw)

        self.logIrPlotWidget = pw = pg.PlotWidget()
        pw.showGrid(True, True, 0.8)
        pw.setLabel('left', 'Log Inverse IR')
        self.logIrPlot = pw.plot(pen=(0, 255, 100))
        splitter.addWidget(pw)

        self.correctionPlotWidget = pw = pg.PlotWidget()
        pw.showGrid(True, True, 0.8)
        pw.setLabel('left', 'Correction Factor')
        self.correctionPlot = pw.plot(
            pen=(255, 255, 200), fillLevel=0, fillBrush=(255, 0, 0, 100))
        splitter.addWidget(pw)

        axes = {ori: Axis(ori) for ori in ['bottom', 'left']}
        for ax in axes.values():
            ax.setGrid(255)
        self.simPlotWidget = pw = pg.PlotWidget(axisItems=axes)
        pw.showGrid(True, True, 0.8)
        pw.setLabel('left', 'Corrected Spectrum')
        self.simPlot = pg.PlotDataItem(pen=(150, 100, 60), stepMode='right')
        pw.addItem(self.simPlot, ignoreBounds=True)
        self.avSimPlot = pw.plot(pen=(255, 255, 200), stepMode='right')
        pw.setLogMode(x=True)
        splitter.addWidget(pw)

        self.msDuration = pg.SpinBox(
            value=50, step=1, bounds=[1, 1000], suffix='ms')
        self.msDuration.sigValueChanging.connect(self.plot)
        self.dbRange = pg.SpinBox(
            value=24, step=1, bounds=[0, 100], suffix='dB')
        self.dbRange.sigValueChanging.connect(self.plot)
        self.kaiserBeta = pg.SpinBox(
            value=5, step=1, bounds=[0, 100])
        self.irSmoothing = pg.SpinBox(
            value=15, step=1, bounds=[0, 30])
        self.irSmoothing.sigValueChanging.connect(self.plot)
        self.kaiserBeta.sigValueChanging.connect(self.plot)
        self.useRefBox = qt.QCheckBox('Use reference')
        self.useRefBox.stateChanged.connect(self.plot)
        exportButton = qt.QPushButton('Export as WAV')
        exportButton.setShortcut('E')
        exportButton.setToolTip('<Key E>')
        exportButton.clicked.connect(self.saveIR)

        hbox = qt.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(qt.QLabel('Duration: '))
        hbox.addWidget(self.msDuration)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Range: '))
        hbox.addWidget(self.dbRange)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Tapering: '))
        hbox.addWidget(self.kaiserBeta)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Smoothing: '))
        hbox.addWidget(self.irSmoothing)
        hbox.addSpacing(32)
        hbox.addWidget(self.useRefBox)
        hbox.addSpacing(32)
        hbox.addWidget(exportButton)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        return topWidget

    def createSharedControls(self) -> qt.QWidget:
        topWidget = qt.QWidget()
        vbox = qt.QVBoxLayout()
        topWidget.setLayout(vbox)

        self.buttons = buttons = qt.QButtonGroup()
        buttons.setExclusive(True)
        spectrumButton = qt.QRadioButton('Spectrum')
        irButton = qt.QRadioButton('Impulse Response')
        buttons.addButton(spectrumButton, 0)
        buttons.addButton(irButton, 1)
        spectrumButton.setChecked(True)
        buttons.idClicked.connect(self.stack.setCurrentIndex)

        screenshotButton = qt.QPushButton('Screenshot')
        screenshotButton.setShortcut('S')
        screenshotButton.setToolTip('<Key S>')
        screenshotButton.clicked.connect(self.screenshot)

        pauseButton = qt.QPushButton('Pause')
        pauseButton.setShortcut('Space')
        pauseButton.setToolTip('<Space>')
        pauseButton.setFocusPolicy(qtgui.Qt.NoFocus)
        pauseButton.clicked.connect(self.setPaused)

        exitButton = qt.QPushButton('Exit')
        exitButton.setShortcut('Esc')
        exitButton.setToolTip('<Esc>')
        exitButton.clicked.connect(self.close)

        hbox = qt.QHBoxLayout()
        hbox.addWidget(spectrumButton)
        hbox.addSpacing(32)
        hbox.addWidget(irButton)
        hbox.addStretch(1)
        hbox.addWidget(screenshotButton)
        hbox.addSpacing(32)
        hbox.addWidget(pauseButton)
        hbox.addSpacing(32)
        hbox.addWidget(exitButton)
        vbox.addLayout(hbox)

        return topWidget


class Axis(pg.AxisItem):

    def logTickStrings(self, values, scale, spacing):
        return [pg.siFormat(10 ** v).replace(' ', '') for v in values]


async def wrap_coro(coro):
    try:
        await coro
    except asyncio.CancelledError:
        pass
    except Exception:
        logging.getLogger('hifiscan').exception('Error in task:')


def main():
    _ = qt.QApplication(sys.argv)
    app = App()
    app.run()


if __name__ == '__main__':
    main()
