"""
    optimization_centralized.duals['active_power_vector_maximum_constraint'].transpose()
    timestep                    2017-01-02 15:00:00  2017-01-02 16:00:00
    node
    (fixed_ev_charger, 4_3)                0.000000             0.000000
    (fixed_ev_charger, 6_8)                0.000000             0.000000
    (fixed_ev_charger, 6_9)                0.000000             0.000000
    (fixed_generator, 4_6)                 0.000000             0.000000
    (fixed_load, 4_1)                      0.000000             0.000000
    (fixed_load, 6_1)                      0.000000             0.000000
    (fixed_load, 6_2)                      0.000000             0.000000
    (fixed_load, 6_3)                      0.000000             0.000000
    (flexible_generator, 4_5)             79.320358            77.640350
    (flexible_generator, 6_10)            78.149059            76.492389
    (flexible_load, 4_2)                   0.000000             0.000000
    (flexible_load, 6_4)                   0.000000             0.000000
    optimization_centralized.duals['reactive_power_vector_maximum_constraint'].transpose()
    timestep                    2017-01-02 15:00:00  2017-01-02 16:00:00
    node
    (fixed_ev_charger, 4_3)                     0.0                  0.0
    (fixed_ev_charger, 6_8)                     0.0                  0.0
    (fixed_ev_charger, 6_9)                     0.0                  0.0
    (fixed_generator, 4_6)                      0.0                  0.0
    (fixed_load, 4_1)                           0.0                  0.0
    (fixed_load, 6_1)                           0.0                  0.0
    (fixed_load, 6_2)                           0.0                  0.0
    (fixed_load, 6_3)                           0.0                  0.0
    (flexible_generator, 4_5)                   0.0                  0.0
    (flexible_generator, 6_10)                  0.0                  0.0
    (flexible_load, 4_2)                        0.0                  0.0
    (flexible_load, 6_4)                        0.0                  0.0
    optimization_centralized.duals['active_power_vector_minimum_constraint'].transpose()
    timestep                    2017-01-02 15:00:00  2017-01-02 16:00:00
    node
    (fixed_ev_charger, 4_3)                0.000000             0.000000
    (fixed_ev_charger, 6_8)                0.000000             0.000000
    (fixed_ev_charger, 6_9)                0.000000             0.000000
    (fixed_generator, 4_6)                 0.000000             0.000000
    (fixed_load, 4_1)                      0.000000             0.000000
    (fixed_load, 6_1)                      0.000000             0.000000
    (fixed_load, 6_2)                      0.000000             0.000000
    (fixed_load, 6_3)                      0.000000             0.000000
    (flexible_generator, 4_5)              0.000000             0.000000
    (flexible_generator, 6_10)             0.000000             0.000000
    (flexible_load, 4_2)                  50.592215            49.584210
    (flexible_load, 6_4)                   2.496844             2.447096
    optimization_centralized.duals['reactive_power_vector_minimum_constraint'].transpose()
    timestep                    2017-01-02 15:00:00  2017-01-02 16:00:00
    node
    (fixed_ev_charger, 4_3)                     0.0                  0.0
    (fixed_ev_charger, 6_8)                     0.0                  0.0
    (fixed_ev_charger, 6_9)                     0.0                  0.0
    (fixed_generator, 4_6)                      0.0                  0.0
    (fixed_load, 4_1)                           0.0                  0.0
    (fixed_load, 6_1)                           0.0                  0.0
    (fixed_load, 6_2)                           0.0                  0.0
    (fixed_load, 6_3)                           0.0                  0.0
    (flexible_generator, 4_5)                   0.0                  0.0
    (flexible_generator, 6_10)                  0.0                  0.0
    (flexible_load, 4_2)                        0.0                  0.0
    (flexible_load, 6_4)                        0.0                  0.0


                                                                                                                        ( '4_2',      'state_of_charge'),
                                                                                                                        ( '4_2', 'power_maximum_margin'),
                                                                                                                        ( '4_2', 'power_minimum_margin'),
                                                                                                                        ( '4_2',         'active_power'),
    (        'fixed_load',  '4_1'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '4_2',       'reactive_power'),
    (     'flexible_load',  '4_2'),   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '4_5', 'power_maximum_margin'),
    (  'fixed_ev_charger',  '4_3'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '4_5', 'power_minimum_margin'),
    ('flexible_generator',  '4_5'),   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '4_5',         'active_power'),
    (   'fixed_generator',  '4_6'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '4_5',       'reactive_power'),
    (        'fixed_load',  '6_1'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '6_4',      'state_of_charge'),
    (        'fixed_load',  '6_2'), = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  @      ( '6_4', 'power_maximum_margin'),
    (        'fixed_load',  '6_3'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '6_4', 'power_minimum_margin'),
    (     'flexible_load',  '6_4'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],         ( '6_4',         'active_power'),
    (  'fixed_ev_charger',  '6_8'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ( '6_4',       'reactive_power'),
    (  'fixed_ev_charger',  '6_9'),   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],         ('6_10', 'power_maximum_margin'),
    ('flexible_generator', '6_10')    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]          ('6_10', 'power_minimum_margin'),
                                                                                                                        ('6_10',         'active_power'),
                                                                                                                        ('6_10',       'reactive_power')

            (        'fixed_load',  '4_1'),                              [0.90165043],         0
            (     'flexible_load',  '4_2'),                              [0.        ],         ( '4_2',         'active_power'),
            (  'fixed_ev_charger',  '4_3'),                              [0.91147014],         0
            ('flexible_generator',  '4_5'),                              [0.        ],         ( '4_5',         'active_power'),
            (   'fixed_generator',  '4_6'),                              [0.96      ],         0
            (        'fixed_load',  '6_1'),                              [0.90165043],    +    0
            (        'fixed_load',  '6_2'),                              [0.90165043],         0
            (        'fixed_load',  '6_3'),                              [0.90165043],         0
            (     'flexible_load',  '6_4'),                              [0.        ],         ( '6_4',         'active_power'),
            (  'fixed_ev_charger',  '6_8'),                              [0.91147014],         0
            (  'fixed_ev_charger',  '6_9'),                              [0.91147014],         0
            ('flexible_generator', '6_10')                               [0.        ],         ('6_10',         'active_power'),



















"""