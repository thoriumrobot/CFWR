import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOfIf.List({ @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true), @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true) })
    @EnsuresLTLengthOfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
    public boolean withcondpostconditionfunc2() {
        v1 = value1.length() - 3;
        v2 = value2.length() - 3;
        v3 = value3.length() - 3;
        return true;
    }
}
