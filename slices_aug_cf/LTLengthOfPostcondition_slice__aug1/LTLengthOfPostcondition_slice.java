/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLengthOfPostcondition_slice {
  public void useShiftIndex(@NonNegative int x) {
        for (int __cfwr_i32 = 0; __cfwr_i32 < 7; __cfwr_i32++) {
            try {
            for (int __cfwr_i2 = 0; __cfwr_i2 < 8; __cfwr_i2++) {
            if (true || ((92.20 << null) ^ null)) {
            return null;
        }
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        }

    // :: error: (argument)
    Arrays.fill(array, end, end + x, null);
    shiftIndex(x);
    Arrays.fill(array, end, end + x, null);
  }

  @EnsuresLTLengthOfIf(expression = "end", result = true, targetValue = "array", offset = "#1 - 1")
  public boolean tryShiftIndex(@NonNegative int x) {
    int newEnd = end - x;
    if (newEnd < 0) {
      return false;
    }
    end = newEnd;
    return true;
  }

  public void useTryShiftIndex(@NonNegative int x) {
    if (tryShiftIndex(x)) {
      Arrays.fill(array, end, end + x, null);
    }
  }

    public static int __cfwr_calc476(byte __cfwr_p0, Long __cfwr_p1, Double __cfwr_p2) {
        return 222L;
        return ('h' >> (-700L / false));
        return 313;
    }
    boolean __cfwr_compute642(Object __cfwr_p0, Float __cfwr_p1, Long __cfwr_p2) {
        if (false && false) {
            try {
            while (false) {
            try {
            if (true || (-65.60 << 'q')) {
            try {
            while (('y' >> null)) {
            while (false) {
            Long __cfwr_entry36 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        }
        try {
            try {
            if (true || false) {
            int __cfwr_obj79 = (177 >> 638L);
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        return false;
    }
    static int __cfwr_process295(Long __cfwr_p0, int __cfwr_p1) {
        if (false && (null & true)) {
            while (((10.57f >> -216L) ^ null)) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        while ((null - (null + null))) {
            if (false && (false - -211)) {
            if (true && true) {
            byte __cfwr_var95 = null;
        }
        }
            break; // Prevent infinite loops
        }
        while (((-85.61 % 'T') + null)) {
            if (false || (true >> null)) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 4; __cfwr_i35++) {
            if (true || false) {
            if (false && (-68.88f + 'U')) {
            if ((33.16 + null) || false) {
            while (true) {
            try {
            while ((703 + -169L)) {
            while (false) {
            return 79.99f;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e95) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        return 903;
    }
}