def pytest_addoption(parser):
    parser.addoption("--hf-token", type=str, default=None)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.hf_token
    if "hf_token" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("hf_token", [option_value])
