/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLengthOfPostcondition_slice {
  public void useShiftIndex(@NonNegative int x) {
        if (false && true) {
            if (true && true) {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 10; __cfwr_i26++) {
            try {
            return 48.18;
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        }
        }

    // :: err
        if (((-760 | null) + (false >> false)) && false) {
            short __cfwr_var52 = ((3.82 & 'b') / (715 - 'H'));
        }
or: (argument)
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

    protected boolean __cfwr_handle273(String __cfwr_p0) {
        Integer __cfwr_result15 = null;
        for (int __cfwr_i47 = 0; __cfwr_i47 < 10; __cfwr_i47++) {
            return (-883L ^ (true << 'P'));
        }
        for (int __cfwr_i67 = 0; __cfwr_i67 < 6; __cfwr_i67++) {
            double __cfwr_var91 = -25.23;
        }
        return (405 << null);
    }
}