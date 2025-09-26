/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        while (false) {
            Float __cfwr_node97 = null;
            break; // Prevent infinite loops
        }

    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true)
  @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true)
  @EnsuresLTLengthOfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
  public boolean withcondpostconditionsfunc2() {
    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
    // :: error: (contracts.conditional.postcondition)
    return true;
  }

  @EnsuresLTLengthOf.List({
    @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"),
    @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  })
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionfunc1() {
    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf.List({
    @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true),
    @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true)
  })
  @EnsuresLTLengthOfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
  public boolean withcondpostconditionfunc2() {
    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
    // :: error: (contracts.conditional.postcondition)
    return true;
      public float __cfwr_temp660(char __cfwr_p0) {
        for (int __cfwr_i28 = 0; __cfwr_i28 < 3; __cfwr_i28++) {
            return null;
        }
        while (true) {
            Integer __cfwr_var75 = null;
            break; // Prevent infinite loops
        }
        return null;
        while (false) {
            if (true || false) {
            while (true) {
            try {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 4; __cfwr_i65++) {
            double __cfwr_obj19 = ((-26.87f & null) * 69.82f);
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return (true % (true * 'F'));
    }
    public int __cfwr_process408(boolean __cfwr_p0) {
        if (false || true) {
            double __cfwr_data26 = -99.69;
        }
        Integer __cfwr_result20 = null;
        for (int __cfwr_i3 = 0; __cfwr_i3 < 1; __cfwr_i3++) {
            Float __cfwr_node39 = null;
        }
        return -22;
    }
}
