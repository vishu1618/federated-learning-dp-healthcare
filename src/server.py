import copy


def federated_average(global_model, client_weights):
    new_state_dict = copy.deepcopy(global_model.state_dict())

    for key in new_state_dict.keys():
        new_state_dict[key] = sum(
            client_weights[i][key] for i in range(len(client_weights))
        ) / len(client_weights)

    global_model.load_state_dict(new_state_dict)
    return global_model