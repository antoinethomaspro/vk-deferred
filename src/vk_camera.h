#pragma once

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>


class MyCamera {
public:
	void setPerspectiveProjection(float fovy, float aspect, float near, float far);

	void setViewDirection(glm::vec3 position, glm::vec3 direction, glm::vec3 up = glm::vec3{ 0.f, -1.f, 0.f });

	//when we want the camera to lock up to a specific point in space (we want the object to be at the center of the viewport)
	void setViewTarget(glm::vec3 position, glm::vec3 target, glm::vec3 up = glm::vec3{ 0.f, -1.f, 0.f });

	//specify the orientation of the camera using euler angles
	void setViewYXZ(glm::vec3 position, glm::vec3 rotation);

	const glm::mat4& getProjection() const { return projectionMatrix; }
	const glm::mat4& getView() const { return viewMatrix; }
	const glm::mat4& getInverseView() const { return inverseViewMatrix; }



private:
	glm::mat4 projectionMatrix{ 1.f };
	glm::mat4 viewMatrix{ 1.f };
	glm::mat4 inverseViewMatrix{ 1.f };
};
