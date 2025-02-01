//
//  CameraFeedView.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//



import SwiftUI
import AVFoundation

struct CameraFeedView: View {
    @StateObject private var viewModel = CameraViewModel()
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Camera preview with processed image overlay
                CameraPreviewLayer(viewModel: viewModel)
                    .ignoresSafeArea()
                
                // Status indicators
                VStack {
                    Spacer()
                    HStack {
                        Circle()
                            .fill(viewModel.isConnectedToServer ? Color.green : Color.red)
                            .frame(width: 10, height: 10)
                        
                        Text(viewModel.isConnectedToServer ? "Connected" : "Disconnected")
                            .font(.caption)
                            .foregroundColor(.white)
                    }
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(20)
                    
                    if let error = viewModel.error {
                        Text(error)
                            .foregroundColor(.red)
                            .padding()
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(10)
                    }
                }
            }
        }
        .ignoresSafeArea()
        .onAppear {
            print("CameraFeedView appeared")
            viewModel.startSession()
        }
        .onDisappear {
            print("CameraFeedView disappeared")
            viewModel.stopSession()
        }
    }
}
