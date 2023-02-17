pipeline {
    agent any
    stages {
        stage('Build Model') {
            steps {
                sh 'cd torch_serving'
                sh 'torch-model-archiver --model-name stable-diffusion --version 1.0 --handler stable_diffusion_handler.py --extra-files model.zip -r requirements.txt -f'
                sh 'sudo docker build . -t cr.yandex/crpbhviu3kh3io6cv5n0/sd'
                sh 'sudo docker push cr.yandex/crpbhviu3kh3io6cv5n0/sd'
            }
        }
        stage('Build VK Bot') {
            steps {
                sh 'cd torch_serving'
            }
        }
        stage('Deploy') {
            steps {
                sh 'kubectl delete deployment.apps/designermodel'
                sh 'kubectl delete deployment.apps/vkbot'
                sh 'kubectl apply -f torch_serving/model.yaml'
                sleep 30000;
                sh 'kubectl apply -f vk_bot/bot.yaml'

            }
        }
    }
}