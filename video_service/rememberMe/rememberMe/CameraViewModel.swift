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
        
        @Published var isConnectedToServer = false
        @Published var lastServerResponse: String?
        @Published var error: String?
        
        // Add frame rate control
        private var lastFrameTime: Date?
        private let frameInterval: TimeInterval = 0.5 // Send frame every 0.5 seconds
        
        override init() {
            captureSession = AVCaptureSession()
            super.init()
            setupSession()
        }
        
        private func setupSession() {
            captureSession.sessionPreset = .high // Set high quality
            
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
        }
        
        func stopSession() {
            guard captureSession.isRunning else { return }
            
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession.stopRunning()
            }
        }
        
        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
            guard let currentTime = lastFrameTime else {
                lastFrameTime = Date()
                return
            }
            
            // Check if enough time has passed since the last frame
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
            sendFrameToServer(image: image)
        }
        
    @Published var processedImage: UIImage?
        
        private func sendFrameToServer(image: UIImage) {
            guard let url = URL(string: "https://0a02-132-205-229-214.ngrok-free.app/receive_image") else {
                DispatchQueue.main.async {
                    self.error = "Invalid server URL"
                }
                return
            }
            
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            guard let imageData = image.jpegData(compressionQuality: 0.5) else {
                DispatchQueue.main.async {
                    self.error = "Failed to compress image"
                }
                return
            }
            
            let base64EncodedImage = imageData.base64EncodedString()
            let body: [String: Any] = ["image": base64EncodedImage]
            
            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            } catch {
                DispatchQueue.main.async {
                    self.error = "Failed to serialize JSON: \(error.localizedDescription)"
                }
                return
            }
            
            URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
                        DispatchQueue.main.async {
                            if let error = error {
                                self?.error = "Network error: \(error.localizedDescription)"
                                self?.isConnectedToServer = false
                                return
                            }
                            
                            guard let httpResponse = response as? HTTPURLResponse else {
                                self?.error = "Invalid server response"
                                self?.isConnectedToServer = false
                                return
                            }
                            
                            if httpResponse.statusCode == 200 {
                                self?.isConnectedToServer = true
                                if let data = data,
                                   let jsonResponse = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                                   let processedImageBase64 = jsonResponse["processed_image"] as? String,
                                   let imageData = Data(base64Encoded: processedImageBase64),
                                   let processedUIImage = UIImage(data: imageData) {
                                    self?.processedImage = processedUIImage
                                }
                            } else {
                                self?.error = "Server error: \(httpResponse.statusCode)"
                                self?.isConnectedToServer = false
                            }
                        }
                    }.resume()
                }
            }
