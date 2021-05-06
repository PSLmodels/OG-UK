import numpy as np
import eurostat
import sys
sys.path.append("../")
import demographics


def test_get_fert():
    '''
    Test of function to get fertility rates from data
    '''
    S = 100
    fert_rates = demographics.get_fert(100, 2018, graph=False)
    assert (fert_rates.shape[0] == S)

def test_get_pop_objs():
    """
    Test of the that omega_SS and the last period of omega_path_S are
    close to each other.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    base_yr = 2018

    pop_dict = demographics.get_pop_objs(
        E, S, T, 1, 100, base_yr, False)

    # assert (np.allclose(0, 1))

    assert (np.allclose(pop_dict['omega_SS'], pop_dict['omega'][-1, :]))


# # def test_pop_smooth():
# #     """
# #     Test that population growth rates evolve smoothly.
# #     """
# #     E = 20
# #     S = 80
# #     T = int(round(4.0 * S))
# #     start_year = 2019

# #     pop_dict = demographics.get_pop_objs(
# #         E, S, T, 1, 100, start_year, False)

# #     assert (np.any(np.absolute(pop_dict['omega'][:-1, :] -
# #                                pop_dict['omega'][1:, :]) < 0.0001))
# #     assert (np.any(np.absolute(pop_dict['g_n'][:-1] -
# #                                pop_dict['g_n'][1:]) < 0.0001))


# # def test_imm_smooth():
# #     """
# #     Test that population growth rates evolve smoothly.
# #     """
# #     E = 20
# #     S = 80
# #     T = int(round(4.0 * S))
# #     start_year = 2019

# #     pop_dict = demographics.get_pop_objs(
# #         E, S, T, 1, 100, start_year, False)

# #     assert (np.any(np.absolute(pop_dict['imm_rates'][:-1, :] -
# #                                pop_dict['imm_rates'][1:, :]) < 0.0001))


def test_get_mort():
    """
    Test of function to get mortality rates from data
    * test that get_mort can download Eurostat data to create mort rates data
    * test that get_mort can create mort rates data from hard drive file
    * test that get_mort works for smaller S
    """
    totpers = 100
    mort_rates_100 = np.array(
        [
            3.66591145e-03,
            2.39126801e-04,
            1.00955544e-04,
            1.03971068e-04,
            7.63766898e-05,
            8.63194467e-05,
            8.32479051e-05,
            6.89784533e-05,
            7.12823198e-05,
            5.41771840e-05,
            8.60721534e-05,
            7.81925973e-05,
            7.65738231e-05,
            1.00496234e-04,
            1.18039489e-04,
            1.71664321e-04,
            1.88827340e-04,
            2.47839958e-04,
            3.33883181e-04,
            3.56739554e-04,
            3.47462223e-04,
            3.81248389e-04,
            3.75178749e-04,
            3.64057355e-04,
            4.02716045e-04,
            4.35649818e-04,
            4.20909841e-04,
            4.37157995e-04,
            5.31146450e-04,
            5.38741460e-04,
            5.75313285e-04,
            6.09542249e-04,
            6.19996738e-04,
            7.12571342e-04,
            7.90041820e-04,
            8.21123721e-04,
            8.91103720e-04,
            1.04662636e-03,
            1.01505960e-03,
            1.13994886e-03,
            1.19320029e-03,
            1.26567128e-03,
            1.46810770e-03,
            1.61516785e-03,
            1.70976781e-03,
            1.87301142e-03,
            2.03649433e-03,
            2.21111861e-03,
            2.37980404e-03,
            2.56063962e-03,
            2.70965776e-03,
            2.95312238e-03,
            3.19957793e-03,
            3.51360537e-03,
            3.68021239e-03,
            4.06637763e-03,
            4.48554196e-03,
            5.04243487e-03,
            5.53668790e-03,
            5.83010413e-03,
            6.40140353e-03,
            7.11171117e-03,
            7.81127971e-03,
            8.54924565e-03,
            8.95031835e-03,
            1.02700529e-02,
            1.09237407e-02,
            1.16443736e-02,
            1.30022888e-02,
            1.39727315e-02,
            1.51572381e-02,
            1.85223525e-02,
            1.94389872e-02,
            2.14136644e-02,
            2.44282988e-02,
            2.78966796e-02,
            3.08102119e-02,
            3.34618129e-02,
            3.70430502e-02,
            4.17021256e-02,
            4.70598883e-02,
            5.37683734e-02,
            5.89196021e-02,
            6.83786323e-02,
            7.58697692e-02,
            8.49440348e-02,
            9.81238557e-02,
            1.11514899e-01,
            1.27908538e-01,
            1.44243350e-01,
            1.60693496e-01,
            1.79586854e-01,
            2.01338200e-01,
            2.19798060e-01,
            2.40991676e-01,
            2.59190883e-01,
            2.86956522e-01,
            3.51599788e-01,
            4.48686244e-01,
            1.00000000e00,
        ]
    )
    mort_rates_11 = np.array(
        [
            4.33841507e-04,
            1.48582181e-04,
            3.92557180e-04,
            6.70716628e-04,
            1.37120133e-03,
            2.82781796e-03,
            6.02102766e-03,
            1.37874887e-02,
            3.48242625e-02,
            9.65187708e-02,
            1.00000000e00,
        ]
    )

    # Test that we get the right mort_rates and infmort_rate when reading data
    # from local hard drive path
    mort_rates_1, infmort_rate_1 = demographics.get_mort(
        totpers, 0, 100, graph=False
    )
    assert mort_rates_1.shape[0] == totpers
    assert infmort_rate_1 == 0.003507
    assert np.allclose(mort_rates_1, mort_rates_100)

    # Test that we get the right mort_rates and infmort_rate when downloading
    # data from the internet
    mort_rates_2, infmort_rate_2 = demographics.get_mort(
        totpers, 0, 100, download=True, graph=False
    )
    assert mort_rates_2.shape[0] == totpers
    assert infmort_rate_2 == 0.003507
    assert np.allclose(mort_rates_2, mort_rates_100)

    # Test that we get the right mort_rates and infmort_rate when we use a
    # a smaller number of total model periods
    totpers_small = 11
    mort_rates_3, infmort_rate_3 = demographics.get_mort(
        totpers_small, 0, 100, download=False, graph=False
    )
    assert mort_rates_3.shape[0] == totpers_small
    assert infmort_rate_3 == 0.003507
    assert np.allclose(mort_rates_3, mort_rates_11)

def test_get_imm_resid():
    '''
    Test of function to download and process immigration rates
    '''
    E = 20
    S = 80
    imm_rates = demographics.get_imm_resid(E + S, 0, 100, 2018, graph=False)
    assert (imm_rates.shape[0] == E + S)

# # def test_pop_rebin():
# #     '''
# #     Test of population rebin function
# #     '''
# #     curr_pop_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# #     totpers_new = 5
# #     rebinned_data = demographics.pop_rebin(curr_pop_dist, totpers_new)
# #     assert (rebinned_data.shape[0] == totpers_new)


