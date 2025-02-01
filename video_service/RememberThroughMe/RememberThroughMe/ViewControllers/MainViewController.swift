//
//  MainViewController.swift
//  RememberThroughMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//

import UIKit
import AVFoundation
import Vision

class MainViewController: UIViewController, ScanViewControllerDelegate, UICollectionViewDataSource {
    func didCapture(memory: Memory) {
    }
    
    // MARK: - UI Components
    private let scanButton: UIButton = {
        let button = UIButton()
        button.translatesAutoresizingMaskIntoConstraints = false
        button.setTitle("Scan Object or Person", for: .normal)
        button.backgroundColor = .systemBlue
        button.layer.cornerRadius = 12
        button.titleLabel?.font = .systemFont(ofSize: 18, weight: .medium)
        button.addTarget(self, action: #selector(scanButtonTapped), for: .touchUpInside)
        return button
    }()
    
    private let memoriesCollectionView: UICollectionView = {
        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .vertical
        layout.minimumLineSpacing = 16
        layout.minimumInteritemSpacing = 16
        let cv = UICollectionView(frame: .zero, collectionViewLayout: layout)
        cv.translatesAutoresizingMaskIntoConstraints = false
        cv.backgroundColor = .systemBackground
        cv.register(MemoryCell.self, forCellWithReuseIdentifier: "MemoryCell")
        return cv
    }()
    
    private let voiceAssistantButton: UIButton = {
        let button = UIButton()
        button.translatesAutoresizingMaskIntoConstraints = false
        button.setImage(UIImage(systemName: "waveform.circle.fill"), for: .normal)
        button.tintColor = .systemBlue
        button.addTarget(self, action: #selector(voiceAssistantButtonTapped), for: .touchUpInside)
        return button
    }()
    
    private let reminderButton: UIButton = {
        let button = UIButton()
        button.translatesAutoresizingMaskIntoConstraints = false
        button.setImage(UIImage(systemName: "bell.circle.fill"), for: .normal)
        button.tintColor = .systemBlue
        button.addTarget(self, action: #selector(reminderButtonTapped), for: .touchUpInside)
        return button
    }()
    
    // MARK: - Properties
    private var memories: [Memory] = []
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
        setupCollectionView()
    }
    
    // MARK: - Setup
    private func setupUI() {
        view.backgroundColor = .systemBackground
        title = "Remember Me"
        navigationController?.navigationBar.prefersLargeTitles = true
        
        view.addSubview(scanButton)
        view.addSubview(memoriesCollectionView)
        view.addSubview(voiceAssistantButton)
        view.addSubview(reminderButton)
    }
    
    private func setupCollectionView() {
        memoriesCollectionView.dataSource = self
        memoriesCollectionView.delegate = self
    }
    
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            // Scan Button Constraints
            scanButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            scanButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            scanButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            scanButton.heightAnchor.constraint(equalToConstant: 56),
            
            // Memories Collection View Constraints
            memoriesCollectionView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            memoriesCollectionView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            memoriesCollectionView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            memoriesCollectionView.bottomAnchor.constraint(equalTo: scanButton.topAnchor, constant: -20),
            
            // Voice Assistant Button Constraints
            voiceAssistantButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            voiceAssistantButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            voiceAssistantButton.widthAnchor.constraint(equalToConstant: 44),
            voiceAssistantButton.heightAnchor.constraint(equalToConstant: 44),
            
            // Reminder Button Constraints
            reminderButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            reminderButton.trailingAnchor.constraint(equalTo: voiceAssistantButton.leadingAnchor, constant: -16),
            reminderButton.widthAnchor.constraint(equalToConstant: 44),
            reminderButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    // MARK: - Button Actions
    @objc private func scanButtonTapped() {
        let scanVC = ScanViewController()
        scanVC.delegate = self
        navigationController?.pushViewController(scanVC, animated: true)
    }
    
    @objc private func voiceAssistantButtonTapped() {
        let voiceAssistantVC = VoiceAssistantViewController()
        navigationController?.pushViewController(voiceAssistantVC, animated: true)
    }
    
    @objc private func reminderButtonTapped() {
        let reminderVC = ReminderViewController()
        navigationController?.pushViewController(reminderVC, animated: true)
    }
}

// MARK: - UICollectionViewDelegateFlowLayout Extension
extension MainViewController: UICollectionViewDelegateFlowLayout {
    @objc func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return memories.count
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        guard let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "MemoryCell", for: indexPath) as? MemoryCell else {
            return UICollectionViewCell()
        }
        cell.configure(with: memories[indexPath.row])
        return cell
    }
    
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        let width = (collectionView.bounds.width - 16) / 2
        return CGSize(width: width, height: width * 1.3)
    }
    
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let memory = memories[indexPath.item]
        let detailVC = MemoryDetailViewController(memory: memory)
        guard let navController = self.navigationController else { return }
        navController.pushViewController(detailVC, animated: true)
    }
}
