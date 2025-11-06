function prbs_seq = generate_prbs(N_bits)
    % generate prbs
    % polynomial formula
    shift_reg = ones(1, 9);  % initial state
    prbs_seq = zeros(1, N_bits);
    
    for i = 1:N_bits
        prbs_seq(i) = shift_reg(9);
        feedback = xor(shift_reg(9), shift_reg(5));
        shift_reg = [feedback, shift_reg(1:8)];
    end
end