//
//  CameraPreviewLayer.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//
import SwiftUI
import AVFoundation

struct CameraPreviewLayer: UIViewRepresentable {
    var viewModel: CameraViewModel
    
    func makeUIView(context: Context) -> UIView {
        let containerView = UIView(frame: UIScreen.main.bounds)
        containerView.backgroundColor = .black
        
        // Add camera preview layer
        if let previewLayer = viewModel.videoPreviewLayer {
            previewLayer.frame = containerView.bounds
            previewLayer.videoGravity = .resizeAspectFill
            containerView.layer.addSublayer(previewLayer)
        }
        
        // Add processed image overlay view
        let overlayImageView = UIImageView(frame: containerView.bounds)
        overlayImageView.contentMode = .scaleAspectFill
        overlayImageView.tag = 100  // Tag for identification
        overlayImageView.backgroundColor = .clear
        containerView.addSubview(overlayImageView)
        
        return containerView
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // Update preview layer frame
        if let previewLayer = viewModel.videoPreviewLayer {
            previewLayer.frame = uiView.bounds
        }
        
        // Update overlay image
        if let overlayImageView = uiView.viewWithTag(100) as? UIImageView {
            overlayImageView.frame = uiView.bounds
            
            // Update the image with animation
            if let newImage = viewModel.processedImage {
                UIView.transition(with: overlayImageView,
                                duration: 0.1,
                                options: .transitionCrossDissolve,
                                animations: {
                    overlayImageView.image = newImage
                }, completion: nil)
            }
        }
    }
}
