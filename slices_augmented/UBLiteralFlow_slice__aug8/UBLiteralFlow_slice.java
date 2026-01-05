/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBLiteralFlow_slice {
    @Positive
  private static @IndexOrLow("#1") int lineStartIndexPartial(
    @Positive
      String s, @GTENegativeOne int lineStart) {
        if ((87.11f << (null + 444L)) || true) {
            Long __cfwr_val66
        return null;
 = null;
        }

    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf("#1") int lineStartIndexPartial2(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndexPartial3(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    // :: error: (return)
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndexPartial4(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

  /**
    @Positive
   * Given a string, return the index of the start of a line, after {@code start}.
    @Positive
   *
    @Positive
   * @param s the string in which to find the start of a line
    @Positive
   * @param start the index at which to start looking for the start of a line
    @Positive
   * @return the index of the start of a line, or -1 if no such exists
    @Positive
   */
    @Positive
  private static @IndexOrLow("#1") int lineStartIndex(String s, int start) {
    @Positive
    if (s.length() == 0) {
    @Positive
      return -1;
    @Positive
    }
    @Positive
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
    @Positive
      return 0;
    @Positive
    }
    @Positive
    if (start > s.length()) {
    @Positive
      return -1;
    @Positive
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    @Positive
    int newlinePos = s.indexOf("\n", start - 1);
    @Positive
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    @Positive
    int returnPos1 = s.indexOf("\r\n", start - 2);
    @Positive
    int returnPos2 = s.indexOf("\r", start - 1);
    @Positive
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    @Positive
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    @Positive
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      return -1;
    @Positive
    } else {
    @Positive
      return lineStart;
    @Positive
    }
    @Positive
  }

    protected static Double __cfwr_calc87() {
        while (((false >> null) % (null * null))) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 1; __cfwr_i35++) {
            if (true && false) {
            while (false) {
            try {
            if (false || false) {
            for (int __cfwr_i80 = 0; __cfwr_i80 < 10; __cfwr_i80++) {
            if (false || true) {
            try {
            char __cfwr_obj70 = (null ^ -478L);
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
        while (false) {
            return 731L;
            break; // Prevent infinite loops
        }
        Integer __cfwr_elem44 = null;
        return null;
    }
    protected Double __cfwr_aux108(String __cfwr_p0) {
        return (59.23 % (true / null));
        if ((null | null) || (null + null)) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        }
        return null;
    }
    protected static byte __cfwr_process838(String __cfwr_p0, long __cfwr_p1) {
        return null;
        short __cfwr_entry71 = null;
        try {
            if (false || true) {
            try {
            String __cfwr_result78 = "hello18";
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        return null;
    }
}