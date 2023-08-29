# Example file showing a circle moving on screen
import pygame
import sys
import numpy as np
import cv2
import torch
import treetensor.torch as ttorch
from ding.config import compile_config
from ding.envs.gym_retro import env
from ding.config.SQL.gymretro_airstrikergenesis import cfg
from ding.utils import render
from ding.model import DQN
from ding.model import model_wrap
from ding.policy import SQLPolicy

import offline_data_collector


class InteractiveGame():

    def __init__(
            self,
            cfg,
            model,
            policy,
            checkpoint_path,
            enable_ai_assistance=False,
    ) -> None:

        self.cfg = cfg
        self.model = model
        self.policy = policy
        self.checkpoint_path = checkpoint_path
        self.enable_ai_assistance = enable_ai_assistance

        policy_state_dict = torch.load(file_path, map_location=torch.device("cpu"))
        self.policy.learn_mode.load_state_dict(policy_state_dict)

        def single_env_forward_wrapper(forward_fn, cuda=True):

            forward_fn = model_wrap(forward_fn, wrapper_name='argmax_sample').forward

            def _forward(obs):
                # unsqueeze means add batch dim, i.e. (O, ) -> (1, O)
                obs = ttorch.as_tensor(obs).unsqueeze(0)
                if cuda and torch.cuda.is_available():
                    obs = obs.cuda()
                action = forward_fn(obs)["action"]
                # squeeze means delete batch dim, i.e. (1, A) -> (A, )
                action = action.squeeze(0).detach().cpu().numpy()
                return action

            return _forward

        self.ai_assistance = single_env_forward_wrapper(policy._model, cuda=True)

    def run(self):

        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode([640, 448], pygame.RESIZABLE)
        surface = pygame.display.set_mode((640, 448), pygame.RESIZABLE)

        base_font = pygame.font.Font(None, 32)
        status_rect = pygame.Rect(20, 20, 140, 32)
        AI_rect = pygame.Rect(20, 60, 140, 32)
        Human_rect = pygame.Rect(20, 100, 140, 32)
        trajectory_rect = pygame.Rect(20, 140, 140, 32)
        num_of_AI_time_step = 0
        num_of_human_time_step = 0
        color_normal = pygame.Color(122, 122, 122, 122)
        color_active = pygame.Color('darkblue')
        color_passive = pygame.Color('chartreuse4')
        color = color_passive

        game_env = env(cfg=cfg.env)
        obs = game_env.reset()
        img = render(game_env)

        def transformScaleKeepRatio(image, size):
            iwidth, iheight = image.get_size()
            scale = min(size[0] / iwidth, size[1] / iheight)
            new_size = (round(iwidth * scale), round(iheight * scale))
            scaled_image = pygame.transform.smoothscale(image, new_size)
            image_rect = scaled_image.get_rect(center=(size[0] // 2, size[1] // 2))
            return scaled_image, image_rect

        def AirstrikerGenesis_action_dtype_transform(action):
            if action == 0:
                return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif action == 1:
                return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif action == 2:
                return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif action == 3:
                return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif action == 4:
                return [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif action == 5:
                return [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            else:
                raise ValueError('Invalid action!!')

        def AirstrikerGenesis_action_dtype_inverse_transform(action):
            if action[0] == 1:
                if action[6] == 1:
                    return 4
                elif action[7] == 1:
                    return 5
                else:
                    return 1
            else:
                if action[6] == 1:
                    return 2
                elif action[7] == 1:
                    return 3
                else:
                    return 0

        data_collector = offline_data_collector.interactive_offline_data_collector()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_env.close()
                    pygame.quit()
                    # add current time as file name by using datetime
                    from datetime import datetime
                    now = datetime.now()
                    dt_string = now.strftime("%Y%m%d%H%M%S")
                    data_collector.save(f"./output-{dt_string}.hdf5", save_type="hdf5_v2")
                    #sys.exit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if status_rect.collidepoint(event.pos):
                        self.enable_ai_assistance = not self.enable_ai_assistance

            keys = pygame.key.get_pressed()
            action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
            if keys[pygame.K_z]:
                action = action + np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
            if keys[pygame.K_LEFT]:
                action = action + np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
            if keys[pygame.K_RIGHT]:
                action = action + np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)

            if self.enable_ai_assistance:
                if action.sum() > 0:
                    action = AirstrikerGenesis_action_dtype_inverse_transform(action)
                    controled_by_ai = False
                else:
                    action = self.ai_assistance(obs)
                    controled_by_ai = True
            else:
                action = AirstrikerGenesis_action_dtype_inverse_transform(action)
                controled_by_ai = False

            next_obs, rew, done, info = game_env.step(action)
            data_collector.record(obs, action, rew, next_obs, done, info)

            obs = next_obs

            if done:
                obs = game_env.reset()
                img = render(game_env)
            else:
                img = render(game_env)

            img = np.swapaxes(img, 0, 1)

            if controled_by_ai:
                info['control'] = 'AI'
                num_of_AI_time_step += 1
                color = color_passive
                text_status_surface = base_font.render("AI", True, (255, 255, 255))
            else:
                info['control'] = 'Human'
                num_of_human_time_step += 1
                color = color_active
                text_status_surface = base_font.render("Human", True, (255, 255, 255))

            text_AI_surface = base_font.render(str(num_of_AI_time_step), True, (255, 255, 255))
            text_Human_surface = base_font.render(str(num_of_human_time_step), True, (255, 255, 255))
            text_trajectory_surface = base_font.render(str(len(data_collector.trajectory)), True, (255, 255, 255))

            pygame.draw.rect(screen, color, status_rect)
            pygame.draw.rect(screen, color_passive, AI_rect)
            pygame.draw.rect(screen, color_active, Human_rect)
            pygame.draw.rect(screen, color_normal, trajectory_rect)
            screen.blit(text_status_surface, (status_rect.x + 5, status_rect.y + 5))
            screen.blit(text_AI_surface, (AI_rect.x + 5, AI_rect.y + 5))
            screen.blit(text_Human_surface, (Human_rect.x + 5, Human_rect.y + 5))
            screen.blit(text_trajectory_surface, (trajectory_rect.x + 5, trajectory_rect.y + 5))

            AI_rect.w = max(100, text_AI_surface.get_width() + 10)
            Human_rect.w = max(100, text_Human_surface.get_width() + 10)
            trajectory_rect.w = max(100, text_trajectory_surface.get_width() + 10)

            new_image_w, new_image_h = screen.get_size()
            img = cv2.resize(img, dsize=(new_image_h, 4 * new_image_w // 5), interpolation=cv2.INTER_CUBIC)
            surface = pygame.transform.scale(surface, (4 * new_image_w // 5, new_image_h))
            pygame.pixelcopy.array_to_surface(surface, img)
            screen.blit(surface, (1 * new_image_w // 5, 0))
            pygame.display.flip()
            clock.tick(15)


if __name__ == '__main__':

    cfg = compile_config(cfg, policy=SQLPolicy)
    model = DQN(**cfg.policy.model)
    policy = SQLPolicy(cfg.policy, model=model)
    file_path = "./Airstriker-Genesis-CSQLDiscrete/ckpt/eval.pth.tar"

    interactive_game = InteractiveGame(
        cfg=cfg, model=model, policy=policy, checkpoint_path=file_path, enable_ai_assistance=True
    )

    interactive_game.run()
