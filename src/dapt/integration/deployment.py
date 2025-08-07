"""DAPT Model Deployment Module

Provides deployment utilities for DAPT models:
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Model packaging and versioning
- Deployment configuration generation
- Health checks and monitoring setup
"""

import os
import json
import shutil
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import tarfile
import zipfile
import yaml

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    model_name: str
    country_code: str
    version: str
    model_path: str
    deployment_type: str  # docker, aws, gcp, azure, local
    target_environment: str  # dev, staging, prod
    resources: Dict[str, Any]
    environment_variables: Dict[str, str]
    health_check: Dict[str, Any]
    scaling: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DockerDeployment:
    """Docker deployment utilities"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.dockerfile_template = self._get_dockerfile_template()
        self.docker_compose_template = self._get_docker_compose_template()
    
    def _get_dockerfile_template(self) -> str:
        """Get Dockerfile template"""
        return '''
# DAPT Model Deployment Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model files
COPY models/ ./models/

# Set environment variables
{env_vars}

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run application
CMD ["python", "-m", "src.dapt.cli.main", "serve", "--file", "/app/config.yaml"]
'''
    
    def _get_docker_compose_template(self) -> str:
        """Get docker-compose template"""
        return '''
version: '3.8'

services:
  dapt-api:
    build: .
    ports:
      - "{host_port}:{container_port}"
    environment:
{env_vars}
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '{cpu_limit}'
          memory: {memory_limit}
        reservations:
          cpus: '{cpu_reservation}'
          memory: {memory_reservation}

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  redis_data:
'''
    
    def generate_dockerfile(self, output_dir: Path) -> Path:
        """Generate Dockerfile"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment variables
        env_vars = []
        for key, value in self.config.environment_variables.items():
            env_vars.append(f"ENV {key}={value}")
        
        port = self.config.resources.get("port", 8000)
        
        dockerfile_content = self.dockerfile_template.format(
            env_vars="\n".join(env_vars),
            port=port
        )
        
        dockerfile_path = output_dir / "Dockerfile"
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        return dockerfile_path
    
    def generate_docker_compose(self, output_dir: Path) -> Path:
        """Generate docker-compose.yml"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment variables
        env_vars = []
        for key, value in self.config.environment_variables.items():
            env_vars.append(f"      {key}: {value}")
        
        resources = self.config.resources
        port = resources.get("port", 8000)
        
        compose_content = self.docker_compose_template.format(
            host_port=port,
            container_port=port,
            env_vars="\n".join(env_vars),
            cpu_limit=resources.get("cpu_limit", "2.0"),
            memory_limit=resources.get("memory_limit", "4G"),
            cpu_reservation=resources.get("cpu_reservation", "1.0"),
            memory_reservation=resources.get("memory_reservation", "2G")
        )
        
        compose_path = output_dir / "docker-compose.yml"
        with open(compose_path, 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        return compose_path
    
    def generate_requirements(self, output_dir: Path) -> Path:
        """Generate requirements.txt"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        requirements = [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "pyyaml>=5.4.0",
            "psutil>=5.8.0"
        ]
        
        requirements_path = output_dir / "requirements.txt"
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(requirements))
        
        return requirements_path
    
    def build_image(self, output_dir: Path, tag: str = None) -> str:
        """Build Docker image"""
        if not tag:
            tag = f"dapt-{self.config.country_code}-{self.config.model_name}:{self.config.version}"
        
        try:
            # Build image
            cmd = ["docker", "build", "-t", tag, str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"Successfully built Docker image: {tag}")
            return tag
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build Docker image: {e.stderr}")
    
    def run_container(self, image_tag: str, port: int = None) -> str:
        """Run Docker container"""
        if not port:
            port = self.config.resources.get("port", 8000)
        
        try:
            # Run container
            cmd = [
                "docker", "run", "-d",
                "-p", f"{port}:{port}",
                "--name", f"dapt-{self.config.country_code}-{self.config.model_name}",
                image_tag
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            
            print(f"Successfully started container: {container_id}")
            return container_id
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to run Docker container: {e.stderr}")

class CloudDeployment:
    """Cloud deployment utilities"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_aws_config(self, output_dir: Path) -> Dict[str, Path]:
        """Generate AWS deployment configuration"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # ECS Task Definition
        task_definition = {
            "family": f"dapt-{self.config.country_code}-{self.config.model_name}",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": str(self.config.resources.get("cpu", 1024)),
            "memory": str(self.config.resources.get("memory", 2048)),
            "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
            "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
            "containerDefinitions": [
                {
                    "name": "dapt-api",
                    "image": f"ACCOUNT.dkr.ecr.REGION.amazonaws.com/dapt-{self.config.country_code}:{self.config.version}",
                    "portMappings": [
                        {
                            "containerPort": self.config.resources.get("port", 8000),
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {"name": k, "value": v} for k, v in self.config.environment_variables.items()
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/dapt-{self.config.country_code}",
                            "awslogs-region": "us-west-2",
                            "awslogs-stream-prefix": "ecs"
                        }
                    },
                    "healthCheck": {
                        "command": [
                            "CMD-SHELL",
                            "python -c 'import sys; sys.exit(0)' || exit 1"
                        ],
                        "interval": 30,
                        "timeout": 5,
                        "retries": 3,
                        "startPeriod": 60
                    }
                }
            ]
        }
        
        task_def_path = output_dir / "task-definition.json"
        with open(task_def_path, 'w', encoding='utf-8') as f:
            json.dump(task_definition, f, indent=2)
        files['task_definition'] = task_def_path
        
        # ECS Service Definition
        service_definition = {
            "serviceName": f"dapt-{self.config.country_code}-service",
            "cluster": "dapt-cluster",
            "taskDefinition": f"dapt-{self.config.country_code}-{self.config.model_name}",
            "desiredCount": self.config.scaling.get("min_instances", 1),
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": ["subnet-12345", "subnet-67890"],
                    "securityGroups": ["sg-12345"],
                    "assignPublicIp": "ENABLED"
                }
            },
            "loadBalancers": [
                {
                    "targetGroupArn": "arn:aws:elasticloadbalancing:REGION:ACCOUNT:targetgroup/dapt-tg/1234567890123456",
                    "containerName": "dapt-api",
                    "containerPort": self.config.resources.get("port", 8000)
                }
            ]
        }
        
        service_def_path = output_dir / "service-definition.json"
        with open(service_def_path, 'w', encoding='utf-8') as f:
            json.dump(service_definition, f, indent=2)
        files['service_definition'] = service_def_path
        
        # CloudFormation template
        cf_template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"DAPT {self.config.country_code} model deployment",
            "Resources": {
                "DAPTCluster": {
                    "Type": "AWS::ECS::Cluster",
                    "Properties": {
                        "ClusterName": "dapt-cluster"
                    }
                },
                "DAPTLogGroup": {
                    "Type": "AWS::Logs::LogGroup",
                    "Properties": {
                        "LogGroupName": f"/ecs/dapt-{self.config.country_code}",
                        "RetentionInDays": 7
                    }
                }
            }
        }
        
        cf_path = output_dir / "cloudformation.json"
        with open(cf_path, 'w', encoding='utf-8') as f:
            json.dump(cf_template, f, indent=2)
        files['cloudformation'] = cf_path
        
        return files
    
    def generate_gcp_config(self, output_dir: Path) -> Dict[str, Path]:
        """Generate GCP deployment configuration"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Cloud Run service configuration
        service_config = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": f"dapt-{self.config.country_code}-{self.config.model_name}",
                "annotations": {
                    "run.googleapis.com/ingress": "all"
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/minScale": str(self.config.scaling.get("min_instances", 1)),
                            "autoscaling.knative.dev/maxScale": str(self.config.scaling.get("max_instances", 10)),
                            "run.googleapis.com/cpu-throttling": "false",
                            "run.googleapis.com/memory": f"{self.config.resources.get('memory', 2)}Gi",
                            "run.googleapis.com/cpu": str(self.config.resources.get("cpu", 1))
                        }
                    },
                    "spec": {
                        "containerConcurrency": 100,
                        "containers": [
                            {
                                "image": f"gcr.io/PROJECT_ID/dapt-{self.config.country_code}:{self.config.version}",
                                "ports": [
                                    {
                                        "containerPort": self.config.resources.get("port", 8000)
                                    }
                                ],
                                "env": [
                                    {"name": k, "value": v} for k, v in self.config.environment_variables.items()
                                ],
                                "resources": {
                                    "limits": {
                                        "cpu": str(self.config.resources.get("cpu", 1)),
                                        "memory": f"{self.config.resources.get('memory', 2)}Gi"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": self.config.resources.get("port", 8000)
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 30
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        service_path = output_dir / "cloud-run-service.yaml"
        with open(service_path, 'w', encoding='utf-8') as f:
            yaml.dump(service_config, f, default_flow_style=False)
        files['cloud_run_service'] = service_path
        
        return files
    
    def generate_azure_config(self, output_dir: Path) -> Dict[str, Path]:
        """Generate Azure deployment configuration"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Azure Container Instances configuration
        aci_config = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "containerGroupName": {
                    "type": "string",
                    "defaultValue": f"dapt-{self.config.country_code}-{self.config.model_name}"
                }
            },
            "resources": [
                {
                    "type": "Microsoft.ContainerInstance/containerGroups",
                    "apiVersion": "2021-03-01",
                    "name": "[parameters('containerGroupName')]",
                    "location": "[resourceGroup().location]",
                    "properties": {
                        "containers": [
                            {
                                "name": "dapt-api",
                                "properties": {
                                    "image": f"REGISTRY.azurecr.io/dapt-{self.config.country_code}:{self.config.version}",
                                    "ports": [
                                        {
                                            "port": self.config.resources.get("port", 8000),
                                            "protocol": "TCP"
                                        }
                                    ],
                                    "environmentVariables": [
                                        {"name": k, "value": v} for k, v in self.config.environment_variables.items()
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": self.config.resources.get("cpu", 1),
                                            "memoryInGB": self.config.resources.get("memory", 2)
                                        }
                                    }
                                }
                            }
                        ],
                        "osType": "Linux",
                        "ipAddress": {
                            "type": "Public",
                            "ports": [
                                {
                                    "port": self.config.resources.get("port", 8000),
                                    "protocol": "TCP"
                                }
                            ]
                        }
                    }
                }
            ]
        }
        
        aci_path = output_dir / "azure-container-instances.json"
        with open(aci_path, 'w', encoding='utf-8') as f:
            json.dump(aci_config, f, indent=2)
        files['azure_aci'] = aci_path
        
        return files

class ModelPackager:
    """Package models for deployment"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def create_deployment_package(self, output_dir: Path) -> Path:
        """Create complete deployment package"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        package_name = f"dapt-{self.config.country_code}-{self.config.model_name}-{self.config.version}"
        package_dir = output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Copy model files
        model_dir = package_dir / "models"
        model_dir.mkdir(exist_ok=True)
        
        if Path(self.config.model_path).exists():
            if Path(self.config.model_path).is_dir():
                shutil.copytree(self.config.model_path, model_dir / "model", dirs_exist_ok=True)
            else:
                shutil.copy2(self.config.model_path, model_dir)
        
        # Create deployment configuration
        deployment_config = {
            "model_info": {
                "name": self.config.model_name,
                "country_code": self.config.country_code,
                "version": self.config.version,
                "deployment_type": self.config.deployment_type,
                "created_at": datetime.now().isoformat()
            },
            "api_config": {
                "port": self.config.resources.get("port", 8000),
                "workers": self.config.resources.get("workers", 1),
                "timeout": self.config.resources.get("timeout", 30)
            },
            "environment": self.config.environment_variables,
            "health_check": self.config.health_check,
            "scaling": self.config.scaling
        }
        
        config_path = package_dir / "deployment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_config, f, indent=2, ensure_ascii=False)
        
        # Generate deployment files based on type
        if self.config.deployment_type == "docker":
            docker_deployment = DockerDeployment(self.config)
            docker_deployment.generate_dockerfile(package_dir)
            docker_deployment.generate_docker_compose(package_dir)
            docker_deployment.generate_requirements(package_dir)
        
        elif self.config.deployment_type in ["aws", "gcp", "azure"]:
            cloud_deployment = CloudDeployment(self.config)
            
            if self.config.deployment_type == "aws":
                cloud_deployment.generate_aws_config(package_dir)
            elif self.config.deployment_type == "gcp":
                cloud_deployment.generate_gcp_config(package_dir)
            elif self.config.deployment_type == "azure":
                cloud_deployment.generate_azure_config(package_dir)
        
        # Create README
        readme_content = self._generate_readme()
        readme_path = package_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return package_dir
    
    def _generate_readme(self) -> str:
        """Generate README for deployment package"""
        return f'''
# DAPT Model Deployment Package

## Model Information
- **Model Name**: {self.config.model_name}
- **Country Code**: {self.config.country_code}
- **Version**: {self.config.version}
- **Deployment Type**: {self.config.deployment_type}
- **Target Environment**: {self.config.target_environment}

## Deployment Instructions

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t dapt-{self.config.country_code}:{self.config.version} .
   ```

2. Run the container:
   ```bash
   docker run -p {self.config.resources.get("port", 8000)}:{self.config.resources.get("port", 8000)} dapt-{self.config.country_code}:{self.config.version}
   ```

3. Or use docker-compose:
   ```bash
   docker-compose up -d
   ```

### Model Usage

Once deployed, the model can be used through the CLI interface:

```bash
# Run prediction
python -m dapt.cli predict --text "مرحبا بكم في دولة الإمارات العربية المتحدة" --country {self.config.country_code}

# Batch prediction
python -m dapt.cli predict-batch --input-file texts.txt --country {self.config.country_code}
```

### Environment Variables

{chr(10).join([f"- `{k}`: {v}" for k, v in self.config.environment_variables.items()])}

### Resource Requirements

- CPU: {self.config.resources.get("cpu", "1 core")}
- Memory: {self.config.resources.get("memory", "2GB")}
- Port: {self.config.resources.get("port", 8000)}

### Scaling Configuration

- Min Instances: {self.config.scaling.get("min_instances", 1)}
- Max Instances: {self.config.scaling.get("max_instances", 10)}
- Target CPU: {self.config.scaling.get("target_cpu", 70)}%

### Health Check

The service includes basic health checks:
- Interval: {self.config.health_check.get("interval", 30)}s
- Timeout: {self.config.health_check.get("timeout", 5)}s
- Retries: {self.config.health_check.get("retries", 3)}

## Support

For issues and support, please refer to the DAPT documentation.
'''
    
    def create_archive(self, package_dir: Path, format: str = "tar.gz") -> Path:
        """Create archive of deployment package"""
        archive_name = f"{package_dir.name}.{format}"
        archive_path = package_dir.parent / archive_name
        
        if format == "tar.gz":
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(package_dir, arcname=package_dir.name)
        elif format == "zip":
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in package_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir.parent)
                        zip_file.write(file_path, arcname)
        else:
            raise ValueError(f"Unsupported archive format: {format}")
        
        return archive_path

def create_deployment_config(
    model_name: str,
    country_code: str,
    model_path: str,
    deployment_type: str = "docker",
    target_environment: str = "prod",
    **kwargs
) -> DeploymentConfig:
    """Create deployment configuration"""
    
    # Default configuration
    default_config = {
        "version": "1.0.0",
        "resources": {
            "cpu": 2,
            "memory": 4,
            "port": 8000,
            "workers": 1,
            "timeout": 30
        },
        "environment_variables": {
            "PYTHONPATH": "/app",
            "MODEL_PATH": "/app/models",
            "LOG_LEVEL": "INFO"
        },
        "health_check": {
            "interval": 30,
            "timeout": 5,
            "retries": 3
        },
        "scaling": {
            "min_instances": 1,
            "max_instances": 10,
            "target_cpu": 70
        }
    }
    
    # Merge with provided kwargs
    for key, value in kwargs.items():
        if key in default_config and isinstance(default_config[key], dict):
            default_config[key].update(value)
        else:
            default_config[key] = value
    
    return DeploymentConfig(
        model_name=model_name,
        country_code=country_code,
        model_path=model_path,
        deployment_type=deployment_type,
        target_environment=target_environment,
        **default_config
    )

def deploy_model(
    model_path: str,
    country_code: str,
    model_name: str = None,
    deployment_type: str = "docker",
    output_dir: str = "./deployment",
    **kwargs
) -> Path:
    """Deploy model with specified configuration"""
    
    if not model_name:
        model_name = f"model_{country_code}"
    
    # Create deployment configuration
    config = create_deployment_config(
        model_name=model_name,
        country_code=country_code,
        model_path=model_path,
        deployment_type=deployment_type,
        **kwargs
    )
    
    # Create deployment package
    packager = ModelPackager(config)
    package_dir = packager.create_deployment_package(Path(output_dir))
    
    print(f"Deployment package created: {package_dir}")
    
    # Build Docker image if requested
    if deployment_type == "docker" and kwargs.get("build_image", False):
        docker_deployment = DockerDeployment(config)
        image_tag = docker_deployment.build_image(package_dir)
        print(f"Docker image built: {image_tag}")
    
    return package_dir

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python deployment.py <model_path> <country_code> <model_name> [deployment_type]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    country_code = sys.argv[2]
    model_name = sys.argv[3]
    deployment_type = sys.argv[4] if len(sys.argv) > 4 else "docker"
    
    deploy_model(
        model_path=model_path,
        country_code=country_code,
        model_name=model_name,
        deployment_type=deployment_type,
        build_image=True
    )