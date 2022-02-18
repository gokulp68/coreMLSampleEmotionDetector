//
//  ViewController.swift
//  swiftML
//
//  Created by Gokul P on 17/02/22.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {
    
    lazy var predictionLable: UILabel = {
        var label = UILabel()
        label.numberOfLines = 0
        label.textAlignment = .center
        label.text = "Let's check your Emotion :)"
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    lazy var openCameraButton: UIButton = {
        var configuration = UIButton.Configuration.filled()
        configuration.cornerStyle = .medium
        configuration.baseBackgroundColor = .red
        configuration.baseForegroundColor = UIColor.white
        configuration.buttonSize = .large
        configuration.title = "open Camera"
        let button = UIButton(configuration: configuration, primaryAction: nil)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    lazy var openGalleryButton: UIButton = {
        var configuration = UIButton.Configuration.filled()
        configuration.cornerStyle = .medium
        configuration.baseBackgroundColor = .red
        configuration.baseForegroundColor = UIColor.white
        configuration.buttonSize = .large
        configuration.title = "Open Gallery"
        let button = UIButton(configuration: configuration, primaryAction: nil)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    lazy var buttonStackView: UIStackView = {
       let stackView = UIStackView()
        stackView.translatesAutoresizingMaskIntoConstraints = false
        stackView.spacing = 10
        return stackView
    }()

    lazy var imageView: UIImageView = {
        let iV = UIImageView()
        iV.translatesAutoresizingMaskIntoConstraints = false
        return iV
    }()
    
    lazy var imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBlue
        addViews()
        setButtonsActions()
    }
    
    // MARK: - view managmant
    
    private func addViews() {
        
        //Buttons
        view.addSubview(buttonStackView)
        buttonStackView.addArrangedSubview(openCameraButton)
        buttonStackView.addArrangedSubview(openGalleryButton)
        openCameraButton.heightAnchor.constraint(equalToConstant: 30).isActive = true
        openGalleryButton.heightAnchor.constraint(equalToConstant: 30).isActive = true
        buttonStackView.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        buttonStackView.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -90).isActive = true
        
        //ImageView
        view.addSubview(imageView)
        imageView.widthAnchor.constraint(equalToConstant: 300).isActive = true
        imageView.heightAnchor.constraint(equalToConstant: 450).isActive = true
        imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        imageView.centerYAnchor.constraint(equalTo: view.centerYAnchor).isActive = true
        
        //Prediction label
        view.addSubview(predictionLable)
        predictionLable.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        predictionLable.bottomAnchor.constraint(equalTo: imageView.topAnchor, constant: -30).isActive = true
        
    }
    
    
    //MARK: - Button actions
    
    private func setButtonsActions() {
        openGalleryButton.addAction( UIAction(handler: { [weak self] _ in
            guard let self = self else { return }
            self.predictionLable.text = "Let's check your Emotion :)"
            self.openGallery()
        }), for: .touchUpInside)
        
        openCameraButton.addAction( UIAction(handler: { [weak self] _ in
            guard let self = self else { return }
            self.openCamera()
            self.predictionLable.text = "Let's check your Emotion :)"
        }), for: .touchUpInside)
    }
    
    private func openGallery() {
        if UIImagePickerController.isSourceTypeAvailable(.savedPhotosAlbum) {
            imagePicker.delegate = self
            imagePicker.sourceType = .savedPhotosAlbum
            imagePicker.allowsEditing = false
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    private func openCamera() {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            imagePicker.sourceType = .camera
            imagePicker.allowsEditing = true
            imagePicker.delegate = self
            present(imagePicker, animated: true)
        }
    }
}

//MARK: - image picker delegates
extension ViewController: UIImagePickerControllerDelegate & UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage, let cgImage = pickedImage.cgImage {
            imageView.contentMode = .scaleAspectFit
            imageView.image = pickedImage
            detectObjects(from: CIImage(cgImage: cgImage))
        }
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
}

//MARK: - ML Model handlings
extension ViewController {
    
    func detectObjects(from image: CIImage) {
        predictionLable.text = "Hmm I'm Looking..."
        guard let model = try? CNNEmotions(configuration: MLModelConfiguration()).model else { return }
        let visionModel = generateVisionModel(from: model)
        let visionRequest = generateVisionRequest(with: visionModel)
        handleImageRequest(visionRequest, withImage: image)
    }
    
    func generateVisionModel(from model: MLModel) -> VNCoreMLModel {
      guard let visionModel = try? VNCoreMLModel(for: model) else {
        fatalError("Failed to load CoreML model")
      }
      
      return visionModel
    }
    
    func generateVisionRequest(with model: VNCoreMLModel) -> VNCoreMLRequest {
        let visionRequest = VNCoreMLRequest(model: model) { [weak self] (request, error) in
            if let error = error {
                print(error)
            } else {
                guard let results = request.results as? [VNClassificationObservation],
                    let topResult = results.first else {
                    fatalError("Unexpected result type from VNCoreMLRequest")
                }
                DispatchQueue.main.async {
                    self?.predictionLable.text = "U seems really \(topResult.identifier) \n I'm saying with confidence \(topResult.confidence)"
                }
                print("Prediction: \(topResult.identifier). Confidence: \(topResult.confidence)")
            }
        }

        return visionRequest
    }
    
    func handleImageRequest(_ request: VNCoreMLRequest, withImage image: CIImage) {

        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                print(error)
            }
        }
    }
}
