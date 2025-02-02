//
//  CameraPreviewLayer.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//
import SwiftUI
import AVFoundation

struct CameraPreviewLayer: UIViewRepresentable {
    @ObservedObject var viewModel: CameraViewModel
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black
        
        // Only create the processed image view
        let imageView = UIImageView(frame: view.bounds)
        imageView.tag = 100
        imageView.contentMode = .scaleAspectFill
        imageView.backgroundColor = .black
        view.addSubview(imageView)
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if let imageView = uiView.viewWithTag(100) as? UIImageView {
            imageView.frame = uiView.bounds
            imageView.image = viewModel.processedImage
        }
    }
}
