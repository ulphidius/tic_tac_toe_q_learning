const GRID_SIZE: usize = 3;

const REWARD_STANDARD: i8 = 0;
const REWARD_IMPOSSIBLE: i8 = -10;
const REWARD_WIN: i8 = 5;
const REWARD_LOSE: i8 = -5;
const LEARNING_RATE: f32 = 0.3;
const DISCOUNT_FACTOR: f32 = 0.9;

#[derive(Debug, PartialEq)]
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

    pub fn player_win(&self, player: bool) -> bool {
        let grid_max_index = GRID_SIZE - 1;
        let mut horizontal = true;
        let mut vertical = true;
        let mut diagonal_left_rigth = true;
        let mut diagonal_rigth_left = true;

        let is_player = |original_state: bool, grid_element: Option<bool>, player: bool| {
            return match grid_element {
                None => false,
                Some(value) => {
                    if value == player {
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
}

struct Agent {
    cumulated_reward: i32,
    policy: Policy
}

struct Policy {
    q_table: Vec<Vec<i32>>
}

impl Policy {
    pub fn new() -> Self {
        return Policy {
            q_table: Vec::new()
        }
    }

    // #Q(st, at) = Q(st, at) + learning_rate * (reward + discount_factor * max(Q(state)) - Q(st, at))
    pub fn update() {
        unimplemented!();
    }

    pub fn get_best_action() {
        unimplemented!();
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
        let mut sample_data = Environment {
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

        assert_eq!(sample_data1.player_win(player1), true);
        assert_eq!(sample_data1.player_win(player2), false);

        assert_eq!(sample_data2.player_win(player2), true);
        assert_eq!(sample_data2.player_win(player1), false);
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

        assert_eq!(sample_data1.player_win(player1), true);
        assert_eq!(sample_data1.player_win(player2), false);

        assert_eq!(sample_data2.player_win(player2), true);
        assert_eq!(sample_data2.player_win(player1), false);
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

        assert_eq!(sample_data1.player_win(player1), true);
        assert_eq!(sample_data1.player_win(player2), false);

        assert_eq!(sample_data2.player_win(player2), true);
        assert_eq!(sample_data2.player_win(player1), false);
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

        assert_eq!(sample_data1.player_win(player1), true);
        assert_eq!(sample_data1.player_win(player2), false);

        assert_eq!(sample_data2.player_win(player2), true);
        assert_eq!(sample_data2.player_win(player1), false);
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

        assert_eq!(sample_data1.player_win(true), false);
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
        
        assert_eq!(sample_data1.player_win(true), false);
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

        assert_eq!(sample_data1.player_win(true), false);
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
        
        assert_eq!(sample_data1.player_win(true), false);
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

        assert_eq!(sample_data1.player_win(true), false);
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
        
        assert_eq!(sample_data1.player_win(true), false);
    }
}
