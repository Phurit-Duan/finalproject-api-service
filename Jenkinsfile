pipeline {

    agent {
        label 'master'
    }

    environment {
        image = "phuritmurin/django-find-objects"
        registry = "docker.io"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Print Environment') {
            steps {
                sh('ls -al')
                sh('printenv')
                sh "docker system prune -f --all"
            }
        }
        
        stage('Build docker image') {
            steps {
                script {
                    docker.withRegistry('', 'dockerhub') {
                        def slackImage = docker.build("${env.image}:${BUILD_NUMBER}")
                        slackImage.push()
                        slackImage.push('latest')
                    }
                }
            }
        }

        stage('Deployment'){
            steps {
                sh "docker-compose up -d"
                sh "docker system prune -f --all"
            }
            
        }
        
    }
}

