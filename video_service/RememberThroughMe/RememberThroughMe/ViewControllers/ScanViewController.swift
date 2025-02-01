//
//  ScanViewController.swift
//  RememberThroughMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//

import UIKit
import AVFoundation
import Vision

protocol ScanViewControllerDelegate: AnyObject {
    func didCapture(memory: Memory)
}

class ScanViewController: UIViewController {
    weak var delegate: ScanViewControllerDelegate?
    
    private let captureSession = AVCaptureSession()
    private let previewLayer = AVCaptureVideoPreviewLayer()
    
    private let captureButton: UIButton = {
        let button = UIButton()
        button.translatesAutoresizingMaskIntoConstraints = false
        button.setImage(UIImage(systemName: "camera.circle.fill"), for: .normal)
        button.tintColor = .white
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 2)
        button.layer.shadowOpacity = 0.5
        return button
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupUI()
    }
    
    private func setupCamera() {
        // Basic camera setup - you'll need to implement full camera functionality
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.bounds
    }
    
    private func setupUI() {
        view.addSubview(captureButton)
        
        NSLayoutConstraint.activate([
            captureButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -30),
            captureButton.widthAnchor.constraint(equalToConstant: 80),
            captureButton.heightAnchor.constraint(equalToConstant: 80)
        ])
    }
}
