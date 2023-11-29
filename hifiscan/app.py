import asyncio
import copy
import datetime as dt
import logging
import os
import pickle
import signal
import sys
from pathlib import Path

from PyQt6 import QtCore as qtcore, QtGui as qtgui, QtWidgets as qt
import numpy as np
import pyqtgraph as pg
import sounddevice as sd

import hifiscan as hifi


class App(qt.QWidget):

    SAMPLE_RATES = {rate: i for i, rate in enumerate([
        8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000,
        176400, 192000, 352800, 384000])}

    def __init__(self):
        super().__init__()
        self.paused = False
        self.analyzer = None
        self.refAnalyzer = None
        self.calibration = None
        self.target = None
        self.saveDir = Path.home()
        self.loop = asyncio.get_event_loop_policy().get_event_loop()
        self.task = self.loop.create_task(wrap_coro(self.analyze()))

        self.stack = qt.QStackedWidget()
        self.stack.addWidget(self.spectrumWidget())
        self.stack.addWidget(self.irWidget())
        self.stack.currentChanged.connect(self.plot)

        vbox = qt.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.stack)
        vbox.addWidget(self.sharedControls())

        self.resetAudio()
        self.setLayout(vbox)
        self.setWindowTitle('HiFi Scan')
        self.resize(1800, 900)
        self.show()

    async def analyze(self):
        while True:
            self.audioChanged = False
            try:
                rate = int(self.rateCombo.currentText())
                audio = None
                audio = hifi.Audio(rate)
                while not self.audioChanged:
                    lo = self.lo.value()
                    hi = self.hi.value()
                    secs = self.secs.value()
                    ampl = self.ampl.value() / 100
                    ch = self.channelCombo.currentIndex()
                    if self.paused or lo >= hi or secs <= 0 or not ampl:
                        await asyncio.sleep(0.1)
                        continue

                    analyzer = hifi.Analyzer(lo, hi, secs, audio.rate, ampl,
                                             self.calibration, self.target)
                    sound = analyzer.chirp
                    if ch:
                        silence = np.zeros_like(sound)
                        sound = [sound, silence] if ch == 1 \
                            else [silence, sound]
                    audio.play(sound)
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
            except Exception as exc:
                qt.QMessageBox.critical(self, 'Error', str(exc))
                self.resetAudio()
            finally:
                if audio:
                    audio.close()

    def resetAudio(self):
        defaultDevice = next((dev for dev in sd.query_devices()
                             if dev['name'] == 'default'), None)
        defaultRate = defaultDevice.get('default_samplerate', 0) \
            if defaultDevice else 0
        if not defaultRate:
            defaultRate = sd.default.samplerate
        if defaultRate not in self.SAMPLE_RATES:
            defaultRate = 48000
        index = self.SAMPLE_RATES[defaultRate]
        self.rateCombo.setCurrentIndex(index)
        self.audioChanged = True

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
            target = self.analyzer.targetSpectrum(spectrum)
            if target:
                self.targetPlot.setData(*target)
            else:
                self.targetPlot.clear()
        if self.refAnalyzer:
            spectrum = self.refAnalyzer.spectrum(smoothing)
            self.refSpectrumPlot.setData(*spectrum)

    def plotIR(self):
        if self.refAnalyzer and self.useCombo.currentIndex() == 0:
            analyzer = self.refAnalyzer
        else:
            analyzer = self.analyzer
        if not analyzer:
            return
        secs = self.msDuration.value() / 1000
        dbRange = self.dbRange.value()
        beta = self.kaiserBeta.value()
        smoothing = self.irSmoothing.value()
        causality = self.causality.value() / 100

        t, ir = analyzer.h_inv(secs, dbRange, beta, smoothing, causality)
        self.irPlot.setData(1000 * t, ir)

        logIr = np.log10(1e-8 + np.abs(ir))
        self.logIrPlot.setData(1000 * t, logIr)

        corrFactor = analyzer.correctionFactor(ir)
        self.correctionPlot.setData(*corrFactor)

        spectrum, spectrum_resamp = analyzer.correctedSpectrum(corrFactor)
        self.simPlot.setData(*spectrum)
        self.avSimPlot.setData(*spectrum_resamp)
        target = analyzer.targetSpectrum(spectrum)
        if target:
            self.targetSimPlot.setData(*target)
        else:
            self.targetSimPlot.clear()

    def screenshot(self):
        timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        name = f'HiFiScan {timestamp}.png'
        path, _ = qt.QFileDialog.getSaveFileName(
            self, 'Save screenshot', str(self.saveDir / name), 'PNG (*.png)')
        if path:
            self.stack.grab().save(path)
            self.saveDir = Path(path).parent

    def saveIR(self):
        if self.refAnalyzer and self.useCombo.currentIndex() == 0:
            analyzer = self.refAnalyzer
        else:
            analyzer = self.analyzer
        if not analyzer:
            return
        ms = int(self.msDuration.value())
        db = int(self.dbRange.value())
        beta = int(self.kaiserBeta.value())
        smoothing = int(self.irSmoothing.value())
        causality = int(self.causality.value())
        _, irInv = analyzer.h_inv(
            ms / 1000, db, beta, smoothing, causality / 100)

        name = f'IR_{ms}ms_{db}dB_{beta}t_{smoothing}s_{causality}c.wav'
        path, _ = qt.QFileDialog.getSaveFileName(
            self, 'Save inverse impulse response',
            str(self.saveDir / name), 'WAV (*.wav)')
        if path:
            data = np.vstack([irInv, irInv])
            hifi.write_wav(path, data, analyzer.rate)
            self.saveDir = Path(path).parent

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

    def spectrumWidget(self) -> qt.QWidget:
        topWidget = qt.QWidget()
        vbox = qt.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        topWidget.setLayout(vbox)

        axes = {ori: Axis(ori) for ori in
                ['bottom', 'left', 'top', 'right']}
        for ax in axes.values():
            ax.setGrid(200)
        self.spectrumPlotWidget = pw = pg.PlotWidget(axisItems=axes)
        pw.setLabel('left', 'Relative Power [dB]')
        pw.setLabel('bottom', 'Frequency [Hz]')
        pw.setLogMode(x=True)
        self.targetPlot = pw.plot(pen=(255, 0, 0), stepMode='right')
        self.refSpectrumPlot = pw.plot(pen=(255, 100, 0), stepMode='right')
        self.spectrumPlot = pw.plot(pen=(0, 255, 255), stepMode='right')
        self.spectrumPlot.curve.setCompositionMode(
            qtgui.QPainter.CompositionMode.CompositionMode_Plus)
        vbox.addWidget(pw)

        self.lo = pg.SpinBox(
            value=20, step=5, bounds=[5, 40000], suffix='Hz')
        self.hi = pg.SpinBox(
            value=20000, step=100, bounds=[5, 40000], suffix='Hz')
        self.secs = pg.SpinBox(
            value=1.0, step=0.1, bounds=[0.1, 30], suffix='s')
        self.ampl = pg.SpinBox(
            value=40, step=1, bounds=[0, 100], suffix='%')
        self.channelCombo = qt.QComboBox()
        self.channelCombo.addItems(['Stereo', 'Left', 'Right'])
        self.spectrumSmoothing = pg.SpinBox(
            value=15, step=1, bounds=[0, 30])
        self.spectrumSmoothing.sigValueChanging.connect(self.plot)

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
        hbox.addWidget(self.channelCombo)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Smoothing: '))
        hbox.addWidget(self.spectrumSmoothing)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        return topWidget

    def irWidget(self) -> qt.QWidget:
        topWidget = qt.QWidget()
        vbox = qt.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        topWidget.setLayout(vbox)
        splitter = qt.QSplitter(qtcore.Qt.Orientation.Vertical)
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
            ax.setGrid(200)
        self.simPlotWidget = pw = pg.PlotWidget(axisItems=axes)
        pw.showGrid(True, True, 0.8)
        pw.setLabel('left', 'Corrected Spectrum')
        self.simPlot = pg.PlotDataItem(pen=(150, 100, 60), stepMode='right')
        pw.addItem(self.simPlot, ignoreBounds=True)
        self.avSimPlot = pw.plot(pen=(255, 255, 200), stepMode='right')
        self.targetSimPlot = pw.plot(pen=(255, 0, 0), stepMode='right')
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
        self.kaiserBeta.sigValueChanging.connect(self.plot)
        self.irSmoothing = pg.SpinBox(
            value=15, step=1, bounds=[0, 30])
        self.irSmoothing.sigValueChanging.connect(self.plot)

        causalityLabel = qt.QLabel('Causality: ')
        causalityLabel.setToolTip('0% = Zero phase, 100% = Zero latency')
        self.causality = pg.SpinBox(
            value=0, step=5, bounds=[0, 100], suffix='%')
        self.causality.sigValueChanging.connect(self.plot)

        self.useCombo = qt.QComboBox()
        self.useCombo.addItems(['Stored measurements', 'Last measurement'])
        self.useCombo.currentIndexChanged.connect(self.plot)

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
        hbox.addWidget(causalityLabel)
        hbox.addWidget(self.causality)
        hbox.addSpacing(32)
        hbox.addWidget(qt.QLabel('Use: '))
        hbox.addWidget(self.useCombo)
        hbox.addStretch(1)
        hbox.addWidget(exportButton)
        vbox.addLayout(hbox)

        return topWidget

    def stereoTool(self):

        def leftPressed():
            path, _ = qt.QFileDialog.getOpenFileName(
                self, 'Load left channel', str(self.saveDir), 'WAV (*.wav)')
            leftLabel.setText(path)
            self.saveDir = Path(path).parent

        def rightPressed():
            path, _ = qt.QFileDialog.getOpenFileName(
                self, 'Load right channel', str(self.saveDir), 'WAV (*.wav)')
            rightLabel.setText(path)
            self.saveDir = Path(path).parent

        def save():
            try:
                L = hifi.read_wav(leftLabel.text())
                R = hifi.read_wav(rightLabel.text())
                left = L.data[0]
                right = R.data[1 if len(R) > 1 else 0]
                if L.rate != R.rate or left.size != right.size:
                    raise ValueError(
                        'L and R must have same size and rate')
                stereo = [left, right]
            except Exception as e:
                msg = qt.QMessageBox(qt.QMessageBox.Icon.Critical, 'Error',
                                     str(e), parent=dialog)
                msg.exec()
            else:
                path, _ = qt.QFileDialog.getSaveFileName(
                    self, 'Save stereo channels',
                    str(self.saveDir), 'WAV (*.wav)')
                if path:
                    self.saveDir = Path(path).parent
                    hifi.write_wav(path, stereo, L.rate)

        leftLabel = qt.QLabel('')
        leftButton = qt.QPushButton('Load')
        leftButton.pressed.connect(leftPressed)
        rightLabel = qt.QLabel('')
        rightButton = qt.QPushButton('Load')
        rightButton.pressed.connect(rightPressed)
        saveButton = qt.QPushButton('Save')
        saveButton.pressed.connect(save)

        grid = qt.QGridLayout()
        grid.setColumnMinimumWidth(2, 400)
        grid.addWidget(qt.QLabel('Left in: '), 0, 0)
        grid.addWidget(leftButton, 0, 1)
        grid.addWidget(leftLabel, 0, 2)
        grid.addWidget(qt.QLabel('Right in: '), 1, 0)
        grid.addWidget(rightButton, 1, 1)
        grid.addWidget(rightLabel, 1, 2)
        grid.addWidget(qt.QLabel('Stereo out: '), 2, 0)
        grid.addWidget(saveButton, 2, 1, 1, 2)

        dialog = qt.QDialog(self)
        dialog.setWindowTitle('Convert Left + Right to Stereo')
        dialog.setLayout(grid)
        dialog.exec()

    def causalityTool(self):

        def load():
            path, _ = qt.QFileDialog.getOpenFileName(
                self, 'Load Impulse Response',
                str(self.saveDir), 'WAV (*.wav)')
            inputLabel.setText(path)
            self.saveDir = Path(path).parent

        def save():
            caus = causality.value() / 100
            try:
                irIn = hifi.read_wav(inputLabel.text())
                out = [hifi.transform_causality(channel, caus)
                       for channel in irIn.data]
            except Exception as e:
                msg = qt.QMessageBox(qt.QMessageBox.Icon.Critical, 'Error',
                                     str(e), parent=dialog)
                msg.exec()
            else:
                name = Path(inputLabel.text()).stem + \
                    f'_{causality.value():.0f}c.wav'
                path, _ = qt.QFileDialog.getSaveFileName(
                    self, 'Save Impulse Response',
                    str(self.saveDir / name), 'WAV (*.wav)')
                if path:
                    self.saveDir = Path(path).parent
                    hifi.write_wav(path, out, irIn.rate)

        causality = pg.SpinBox(value=0, step=5, bounds=[0, 100], suffix='%')
        inputLabel = qt.QLabel('')
        loadButton = qt.QPushButton('Load')
        loadButton.pressed.connect(load)
        saveButton = qt.QPushButton('Save')
        saveButton.pressed.connect(save)

        grid = qt.QGridLayout()
        grid.setColumnMinimumWidth(2, 400)
        grid.addWidget(qt.QLabel('Input IR: '), 0, 0)
        grid.addWidget(loadButton, 0, 1)
        grid.addWidget(inputLabel, 0, 2)
        grid.addWidget(qt.QLabel('New causality: '), 1, 0)
        grid.addWidget(causality, 1, 1)
        grid.addWidget(qt.QLabel('Output IR: '), 2, 0)
        grid.addWidget(saveButton, 2, 1, 2, 2)

        dialog = qt.QDialog(self)
        dialog.setWindowTitle('Change causality of Impulse Response')
        dialog.setLayout(grid)
        dialog.exec()

    def sharedControls(self) -> qt.QWidget:
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

        def setAudioChanged():
            self.audioChanged = True

        self.rateCombo = qt.QComboBox()
        self.rateCombo.addItems(str(rate) for rate in self.SAMPLE_RATES)
        self.rateCombo.currentIndexChanged.connect(setAudioChanged)

        def toolsPressed():
            tools.popup(toolsButton.mapToGlobal(qtcore.QPoint(0, 0)))

        tools = qt.QMenu()
        tools.addAction('Convert L + R to Stereo', self.stereoTool)
        tools.addAction('Change IR causality', self.causalityTool)

        toolsButton = qt.QPushButton('Tools...')
        toolsButton.pressed.connect(toolsPressed)

        def loadCalibration():
            path, _ = qt.QFileDialog.getOpenFileName(
                self, 'Load mic calibration', str(self.saveDir))
            if path:
                cal = hifi.read_correction(path)
                if cal:
                    self.calibration = cal
                    calAction.setText(calTxt + path)
                    self.saveDir = Path(path).parent
                else:
                    clearCalibration()

        def clearCalibration():
            self.calibration = None
            calAction.setText(calTxt + 'None')

        def loadTarget():
            path, _ = qt.QFileDialog.getOpenFileName(
                self, 'Load target curve', str(self.saveDir))
            if path:
                target = hifi.read_correction(path)
                if target:
                    self.target = target
                    targetAction.setText(targetTxt + path)
                    self.saveDir = Path(path).parent
                else:
                    clearTarget()

        def clearTarget():
            self.target = None
            targetAction.setText(targetTxt + 'None')

        def correctionsPressed():
            corr.popup(correctionsButton.mapToGlobal(qtcore.QPoint(0, 0)))

        calTxt = 'Mic Calibration: '
        targetTxt = 'Target Curve: '
        corr = qt.QMenu()
        calAction = corr.addAction(calTxt + 'None', loadCalibration)
        corr.addAction('Load', loadCalibration)
        corr.addAction('Clear', clearCalibration)
        corr.addSeparator()
        targetAction = corr.addAction(targetTxt + 'None', loadTarget)
        corr.addAction('Load', loadTarget)
        corr.addAction('Clear', clearTarget)

        correctionsButton = qt.QPushButton('Corrections...')
        correctionsButton.pressed.connect(correctionsPressed)

        def storeButtonClicked():
            if self.analyzer:
                if self.analyzer.isCompatible(self.refAnalyzer):
                    self.refAnalyzer.addMeasurements(self.analyzer)
                else:
                    self.refAnalyzer = copy.copy(self.analyzer)
                setMeasurementsText()
                self.plot()

        def clearButtonClicked():
            self.refAnalyzer = None
            self.refSpectrumPlot.clear()
            setMeasurementsText()
            self.plot()

        def setMeasurementsText():
            num = self.refAnalyzer.numMeasurements if self.refAnalyzer else 0
            measurementsLabel.setText(f'Measurements: {num if num else ""}')

        measurementsLabel = qt.QLabel('')
        setMeasurementsText()

        storeButton = qt.QPushButton('Store')
        storeButton.clicked.connect(storeButtonClicked)
        storeButton.setShortcut('S')
        storeButton.setToolTip('<Key S>')

        clearButton = qt.QPushButton('Clear')
        clearButton.clicked.connect(clearButtonClicked)
        clearButton.setShortcut('C')
        clearButton.setToolTip('<Key C>')

        def load():
            path, _ = qt.QFileDialog.getOpenFileName(
                self, 'Load measurements', str(self.saveDir))
            if path:
                with open(path, 'rb') as f:
                    self.refAnalyzer = pickle.load(f)
                setMeasurementsText()
                self.plot()

        def loadStore():
            path, _ = qt.QFileDialog.getOpenFileName(
                self, 'Load and Store measurements', str(self.saveDir))
            if path:
                with open(path, 'rb') as f:
                    analyzer: hifi.Analyzer = pickle.load(f)
                if analyzer and analyzer.isCompatible(self.refAnalyzer):
                    self.refAnalyzer.addMeasurements(analyzer)
                else:
                    self.refAnalyzer = analyzer
                setMeasurementsText()
                self.plot()

        def save():
            analyzer = self.refAnalyzer or self.analyzer
            timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            name = f'Measurements={analyzer.numMeasurements}, {timestamp}'
            path, _ = qt.QFileDialog.getSaveFileName(
                self, 'Save measurements',
                str(self.saveDir / name))
            if path:
                self.saveDir = Path(path).parent
                with open(path, 'wb') as f:
                    pickle.dump(analyzer, f)
                self.plot()

        def filePressed():
            fileMenu.popup(fileButton.mapToGlobal(qtcore.QPoint(0, 0)))

        fileMenu = qt.QMenu()
        fileMenu.addAction('Load', load)
        fileMenu.addAction('Load and Store', loadStore)
        fileMenu.addAction('Save', save)

        fileButton = qt.QPushButton('File...')
        fileButton.clicked.connect(filePressed)

        screenshotButton = qt.QPushButton('Screenshot')
        screenshotButton.clicked.connect(self.screenshot)

        def setPaused():
            self.paused = not self.paused

        pauseButton = qt.QPushButton('Pause')
        pauseButton.setShortcut('Space')
        pauseButton.setToolTip('<Space>')
        pauseButton.setFocusPolicy(qtcore.Qt.FocusPolicy.NoFocus)
        pauseButton.clicked.connect(setPaused)

        exitButton = qt.QPushButton('Exit')
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setToolTip('Ctrl+Q')
        exitButton.clicked.connect(self.close)

        hbox = qt.QHBoxLayout()
        hbox.addWidget(spectrumButton)
        hbox.addSpacing(16)
        hbox.addWidget(irButton)
        hbox.addSpacing(64)
        hbox.addWidget(toolsButton)
        hbox.addSpacing(32)
        hbox.addWidget(correctionsButton)
        hbox.addStretch(1)
        hbox.addWidget(measurementsLabel)
        hbox.addWidget(storeButton)
        hbox.addWidget(clearButton)
        hbox.addWidget(fileButton)
        hbox.addStretch(1)
        hbox.addWidget(qt.QLabel('Sample rate:'))
        hbox.addWidget(self.rateCombo)
        hbox.addStretch(1)
        hbox.addWidget(screenshotButton)
        hbox.addWidget(pauseButton)
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
