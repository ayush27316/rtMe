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
        let view = UIView(frame: UIScreen.main.bounds)
        
        // Make sure preview layer exists and is properly configured
        if let previewLayer = viewModel.videoPreviewLayer {
            previewLayer.frame = view.bounds
            previewLayer.videoGravity = .resizeAspectFill
            previewLayer.connection?.videoOrientation = .portrait
            view.layer.addSublayer(previewLayer)
        } else {
            print("Preview layer is nil!")
        }
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            if let previewLayer = viewModel.videoPreviewLayer {
                previewLayer.frame = uiView.bounds
            }
        }
    }
}
