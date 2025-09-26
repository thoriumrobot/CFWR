/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RepeatLTLengthOfWithError_slice {
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        while ((324L / -277L)) {
            String __cfwr_var75 = "item66";
            break; // Prevent infinite loops
        }

    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true)
  @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true)
  @Ensures
        if (false && (-368 - 539)) {
            while (false) {
            while (false) {
            if (false && false) {
            Boolean __cfwr_elem40 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
LTLengthOfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
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
  }

    private static short __cfwr_helper862(int __cfwr_p0, short __cfwr_p1, String __cfwr_p2) {
        for (int __cfwr_i80 = 0; __cfwr_i80 < 4; __cfwr_i80++) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 1; __cfwr_i67++) {
            short __cfwr_result7 = null;
        }
        }
        if (false || ((-986 ^ -46.02) >> (-17.12f >> -11.29f))) {
            try {
            try {
            try {
            if (false || (91.83 | null)) {
            if (false && false) {
            return null;
        }
        }
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        }
        try {
            if (false && false) {
            if (((66.68 << true) >> -33) || (-88.64 - -664L)) {
            while (true) {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 2; __cfwr_i22++) {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 2; __cfwr_i95++) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        for (int __cfwr_i88 = 0; __cfwr_i88 < 6; __cfwr_i88++) {
            try {
            Float __cfwr_entry25 = null;
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        }
        return null;
    }
    private static Double __cfwr_util124(Character __cfwr_p0) {
        while (((null - 560) % true)) {
            try {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 10; __cfwr_i52++) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 6; __cfwr_i73++) {
            try {
            while (true) {
            short __cfwr_temp89 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        while (false) {
            if (true || true) {
            return 52.77;
        }
            break; // Prevent infinite loops
        }
        Float __cfwr_node33 = null;
        double __cfwr_elem3 = 79.10;
        return null;
    }
    protected Character __cfwr_temp651(Float __cfwr_p0, byte __cfwr_p1) {
        char __cfwr_temp3 = 'E';
        if (false && (44.75f | 'y')) {
            return 160L;
        }
        String __cfwr_result38 = "item91";
        return null;
    }
}