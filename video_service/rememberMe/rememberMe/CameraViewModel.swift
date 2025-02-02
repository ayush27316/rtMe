//
//  CameraViewModel.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//



import SwiftUI
import AVFoundation


@MainActor
class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var captureSession: AVCaptureSession
    var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    var videoDataOutput: AVCaptureVideoDataOutput?
    
    @Published var processedImage: UIImage?
    @Published var faceCount: Int = 0
    @Published var isConnected = false
    @Published var error: String?
    @Published var isConnectedToServer = false
    
    private var lastFrameTime: Date?
    private let frameInterval: TimeInterval = 0.5
    private let webSocketManager = WebSocketManager()
    
    override init() {
        captureSession = AVCaptureSession()
        super.init()
        setupSession()
        setupWebSocket()
    }
    
    private func setupWebSocket() {
            webSocketManager.onReceiveImage = { [weak self] image, faceCount in
                DispatchQueue.main.async {
                    self?.processedImage = image  // This should trigger UI update
                    self?.faceCount = faceCount
                    self?.isConnectedToServer = true
                }
            }
            webSocketManager.connect()
        }
    
    private func setupSession() {
        captureSession.sessionPreset = .high
        
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            error = "Failed to access camera"
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: captureDevice)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput?.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            
            if let videoOutput = videoDataOutput,
               captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
                videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            }
            
            videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            videoPreviewLayer?.videoGravity = .resizeAspectFill
            
        } catch {
            self.error = "Error setting up camera: \(error.localizedDescription)"
        }
    }
    
    func startSession() {
        guard !captureSession.isRunning else { return }
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
        webSocketManager.connect()
    }
    
    func stopSession() {
        guard captureSession.isRunning else { return }
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.stopRunning()
        }
        webSocketManager.disconnect()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let currentTime = lastFrameTime else {
            lastFrameTime = Date()
            return
        }
        
        // Check frame interval
        if Date().timeIntervalSince(currentTime) < frameInterval {
            return
        }
        
        lastFrameTime = Date()
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            DispatchQueue.main.async {
                self.error = "Failed to get pixel buffer"
            }
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            DispatchQueue.main.async {
                self.error = "Failed to create CGImage"
            }
            return
        }
        
        let image = UIImage(cgImage: cgImage)
        webSocketManager.sendImage(image)
    }
}
