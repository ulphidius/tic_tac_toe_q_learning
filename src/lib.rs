use rand::prelude::*;

const GRID_SIZE: usize = 3;
const NUMBER_OF_ACTIONS: usize = 9;
const NUMBER_OF_STATE: u16 = 19683; // 3 pow 9

const REWARD_STANDARD: i8 = 0;
const REWARD_WIN: i8 = 5;
const LEARNING_RATE: f32 = 0.3;
const DISCOUNT_FACTOR: f32 = 0.9;
const EPSILON_STEP: f32 = 0.05;

#[derive(Clone)]
enum Action {
    Collum1Row1,
    Collum1Row2,
    Collum1Row3,
    Collum2Row1,
    Collum2Row2,
    Collum2Row3,
    Collum3Row1,
    Collum3Row2,
    Collum3Row3
}

impl Action {
    pub fn get_value(&self) -> (usize, usize) {
        return match self {
            Action::Collum1Row1 => (0, 0),
            Action::Collum1Row2 => (0, 1),
            Action::Collum1Row3 => (0, 2),
            Action::Collum2Row1 => (1, 0),
            Action::Collum2Row2 => (1, 1),
            Action::Collum2Row3 => (1, 2),
            Action::Collum3Row1 => (2, 0),
            Action::Collum3Row2 => (2, 1),
            Action::Collum3Row3 => (2, 2)
        };
    }

    pub fn get_action(index: (usize, usize)) -> Option<Self> {
        return match index {
            (0, 0) => Some(Action::Collum1Row1),
            (0, 1) => Some(Action::Collum1Row2),
            (0, 2) => Some(Action::Collum1Row3),
            (1, 0) => Some(Action::Collum2Row1),
            (1, 1) => Some(Action::Collum2Row2),
            (1, 2) => Some(Action::Collum2Row3),
            (2, 0) => Some(Action::Collum3Row1),
            (2, 1) => Some(Action::Collum3Row2),
            (2, 2) => Some(Action::Collum3Row3),
            _ => None
        };
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Environment {
    grid: Vec<Vec<Option<bool>>>
}

impl Environment {
    pub fn new() -> Self {
        let mut grid: Vec<Vec<Option<bool>>> = Vec::with_capacity(GRID_SIZE);
        
        for _i in 0..GRID_SIZE {
            grid.push(vec![None; 3]);
        }

        return Environment {
            grid: grid
        };
    }

    pub fn set_value(mut self, new_value: (usize, usize), player: bool) -> Result<Self, &'static str> {
        if self.grid[new_value.0][new_value.1].is_some() {
            return Err("This case is already taken");
        }
        
        self.grid[new_value.0][new_value.1] = Some(player);

        return Ok(self);
    }

    pub fn player_win(&self, player: &bool) -> bool {
        let grid_max_index = GRID_SIZE - 1;
        let mut horizontal = true;
        let mut vertical = true;
        let mut diagonal_left_rigth = true;
        let mut diagonal_rigth_left = true;

        let is_player = |original_state: bool, grid_element: Option<bool>, player: &bool| {
            return match grid_element {
                None => false,
                Some(value) => {
                    if value == *player {
                        return original_state;
                    }
        
                    return false;
                }
            };
            
        };

        for x in 0..GRID_SIZE {
            for y in 0..GRID_SIZE {
                horizontal = is_player(horizontal, self.grid[x][y], player); 
                vertical = is_player(vertical, self.grid[y][x], player);

                diagonal_left_rigth = is_player(diagonal_left_rigth, self.grid[y][y], player);
                diagonal_rigth_left = is_player(diagonal_rigth_left, self.grid[grid_max_index - y][y], player);
            }

            if horizontal || vertical || diagonal_left_rigth || diagonal_rigth_left {
                return true;
            }

        }

        return false;
    }

    // for a state we need a index ho work with a flatten vec
    // index = element_value * number_elements_line
    pub fn get_state_index(&self) -> usize {
        let mut state_index: usize = 0;
        let mut factor = 1;

        for line in self.grid.clone() {
            for element in line {
                state_index += match element {
                    None => 0 * factor,
                    Some(true) => 1 * factor,
                    Some(false) => 2 * factor
                };
            }

            factor *= 3;
        }

        return state_index;
    }

    pub fn get_random_action(&self) -> Action {
        let mut allow_actions: Vec<Action> = Vec::new();
        let mut random = rand::thread_rng();

        for x in 0..self.grid.len() {
            for y in 0..self.grid[x].len() {
                if self.grid[x][y].is_none() {
                    allow_actions.push(Action::get_action((x, y)).unwrap());
                }
            }
        }

        return allow_actions[random.gen_range(0, allow_actions.len())].clone();
    }
}

struct Agent {
    policy: Policy,
    player: bool
}

impl Agent {
    pub fn new() -> Self {
        return Self {
            policy: Policy::new(),
            player: false
        };
    } 
}

struct Policy {
    q_table: Vec<Vec<f32>>,
    epsilon: f32
}

impl Policy {
    pub fn new() -> Self {
        let mut table: Vec<Vec<f32>> = Vec::with_capacity(NUMBER_OF_STATE as usize);
        
        for index in 0..NUMBER_OF_STATE as usize {
            table[index] = vec![0.0; NUMBER_OF_ACTIONS];
        }

        return Policy {
            q_table: table,
            epsilon: 1.0
        }
    }

    // Q(st, at) = Q(st, at) + learning_rate * (reward + discount_factor * max(Q(state) - Q(st, at)))
    // Update the q table with the value of the reward
    // We go with the idea that the input action is posible
    // Get new and previous state_index
    // Get the reward
    // Get q value of previous state
    // Get the best estimated reward for the ne state
    // Update the q table with the equation
    // Return the new the updated Environment and updated policy
    pub fn update(mut self, chose_action: Action, environment: Environment, player: &bool) -> (Policy, Environment) {
        let action_index = chose_action.get_value();

        let previous_state_index = environment.get_state_index();
        let new_state = environment.set_value(action_index, player.clone()).unwrap();

        let new_state_index = new_state.get_state_index();

        let reward = match new_state.player_win(player) {
            true => REWARD_WIN,
            false => REWARD_STANDARD
        };

        let index_collum = action_index.0 + action_index.1 * GRID_SIZE;

        let q_value = self.q_table[previous_state_index][index_collum];
        let mut max_value: f32 = 0.0;

        for value in self.q_table[new_state_index].clone() {
            if max_value < value {
                max_value = value;
            }
        }

        self.q_table[previous_state_index][index_collum] = q_value + LEARNING_RATE + (f32::from(reward) + DISCOUNT_FACTOR * max_value - q_value);

        return (self, new_state);
    }

    // Check if we return random action or not
    // Get the best estimated reward
    // Get the index of the best estimated reward (reverse the get state index equation)
    // Return best action
    pub fn chose_action(&self, environment: &Environment) -> Action {
        let mut random = rand::thread_rng();
        let random_value: f32 = random.gen();

        if random_value < self.epsilon {
            return environment.get_random_action();
        }

        let current_state_index = environment.get_state_index();
        let mut max_value: (usize, f32) = (0, 0.0);

        for index in 0..self.q_table[current_state_index].len() {
            let current_value = self.q_table[current_state_index][index];

            if max_value.1 < current_value {
                max_value = (index, current_value);
            }
        }
    
        let max_value_index = (max_value.0 % GRID_SIZE, max_value.0 / GRID_SIZE);

        return Action::get_action(max_value_index).unwrap();
    }

    // Decrease the random chance after each game
    pub fn update_epsilon(&self) -> f32 {
        return self.epsilon - EPSILON_STEP;
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_set_value_test() {
        let sample_data = Environment::new().set_value((0, 1), true);
        let expected_output: Vec<Vec<Option<bool>>> = vec![
            vec![None, Some(true), None],
            vec![None, None, None],
            vec![None, None, None]
        ];

        assert_eq!(sample_data.unwrap().grid, expected_output);
    }

    #[test]
    #[should_panic]
    fn environment_set_value_index_overflow_test() {
        let sample_data = Environment::new().set_value((0, 10), true);
        
        sample_data.unwrap();
    }

    #[test]
    #[should_panic(expected = "This case is already taken")]
    fn environment_set_value_already_set_test() {
        let sample_data = Environment {
            grid: vec![
                vec![None, Some(true), None],
                vec![None, None, None],
                vec![None, None, None]
            ]
        };

        sample_data.set_value((0, 1), false).unwrap();
    }

    #[test]
    fn environment_is_win_horizontal_test() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), Some(true), Some(true)],
                vec![None, None, None],
                vec![None, None, None]
            ]
        };
        let sample_data2 = Environment {
            grid: vec![
                vec![Some(false), Some(false), Some(false)],
                vec![None, None, None],
                vec![None, None, None]
            ]
        };
        let player1 = true;
        let player2 = false;

        assert_eq!(sample_data1.player_win(&player1), true);
        assert_eq!(sample_data1.player_win(&player2), false);

        assert_eq!(sample_data2.player_win(&player2), true);
        assert_eq!(sample_data2.player_win(&player1), false);
    }

    #[test]
    fn environment_is_win_vertical_test() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), None, None],
                vec![Some(true), None, None],
                vec![Some(true), None, None]
            ]
        };
        let sample_data2 = Environment {
            grid: vec![
                vec![Some(false), None, None],
                vec![Some(false), None, None],
                vec![Some(false), None, None]
            ]
        };
        let player1 = true;
        let player2 = false;

        assert_eq!(sample_data1.player_win(&player1), true);
        assert_eq!(sample_data1.player_win(&player2), false);

        assert_eq!(sample_data2.player_win(&player2), true);
        assert_eq!(sample_data2.player_win(&player1), false);
    }

    #[test]
    fn environment_is_win_diagonal_right_test() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), None, None],
                vec![None, Some(true), None],
                vec![None, None, Some(true)]
            ]
        };
        let sample_data2 = Environment {
            grid: vec![
                vec![Some(false), None, None],
                vec![None, Some(false), None],
                vec![None, None, Some(false)]
            ]
        };
        let player1 = true;
        let player2 = false;

        assert_eq!(sample_data1.player_win(&player1), true);
        assert_eq!(sample_data1.player_win(&player2), false);

        assert_eq!(sample_data2.player_win(&player2), true);
        assert_eq!(sample_data2.player_win(&player1), false);
    }

    #[test]
    fn environment_is_win_diagonal_left_test() {
        let sample_data1 = Environment {
            grid: vec![
                vec![None, None, Some(true)],
                vec![None, Some(true), None],
                vec![Some(true), None, None]
            ]
        };
        let sample_data2 = Environment {
            grid: vec![
                vec![None, None, Some(false)],
                vec![None, Some(false), None],
                vec![Some(false), None, None]
            ]
        };
        let player1 = true;
        let player2 = false;

        assert_eq!(sample_data1.player_win(&player1), true);
        assert_eq!(sample_data1.player_win(&player2), false);

        assert_eq!(sample_data2.player_win(&player2), true);
        assert_eq!(sample_data2.player_win(&player1), false);
    }

    #[test]
    fn environment_is_win_horizontal_incompleted() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), None, Some(true)],
                vec![None, None, None],
                vec![None, None, None]
            ]
        };

        assert_eq!(sample_data1.player_win(&true), false);
    }

    #[test]
    fn environment_is_win_horizontal_cancelled() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), Some(false), Some(true)],
                vec![None, None, None],
                vec![None, None, None]
            ]
        };
        
        assert_eq!(sample_data1.player_win(&true), false);
    }
    
    #[test]
    fn environment_is_win_vertical_incompleted() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), None, None],
                vec![None, None, None],
                vec![Some(true), None, None]
            ]
        };

        assert_eq!(sample_data1.player_win(&true), false);
    }

    #[test]
    fn environment_is_win_vertical_cancelled() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), None, None],
                vec![Some(false), None, None],
                vec![Some(true), None, None]
            ]
        };
        
        assert_eq!(sample_data1.player_win(&true), false);
    }

    #[test]
    fn environment_is_win_diagonal_right_incompleted() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), None, None],
                vec![None, None, None],
                vec![None, None, Some(true)]
            ]
        };

        assert_eq!(sample_data1.player_win(&true), false);
    }

    #[test]
    fn environment_is_win_diagonal_right_cancelled() {
        let sample_data1 = Environment {
            grid: vec![
                vec![Some(true), None, None],
                vec![None, Some(false), None],
                vec![None, None, Some(true)]
            ]
        };
        
        assert_eq!(sample_data1.player_win(&true), false);
    }
}
